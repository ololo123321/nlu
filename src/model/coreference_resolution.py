from typing import Dict, List, Tuple
from abc import abstractmethod
from collections import defaultdict

import tensorflow as tf
import numpy as np

from src.data.base import Example, Arc
from src.model.base import BaseModel, BaseModelBert, ModeKeys
from src.model.layers import StackedBiRNN, GraphEncoder, GraphEncoderInputs, MLP
from src.model.utils import get_dense_labels_from_indices, upper_triangular, get_additive_mask, get_padded_coords_3d, get_span_indices
from src.metrics import classification_report
from src.utils import batches_gen


class BaseModeCoreferenceResolution(BaseModel):
    coref_scope = "coref"

    def __init__(self, sess: tf.Session = None, config: Dict = None):
        super().__init__(sess=sess, config=config)

        # PLACEHOLDERS
        self.mention_coords_ph = None

        # LAYERS
        self.birnn = None

        # TENSORS
        self.logits_train = None
        self.logits_inference = None

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.coref_scope):
            self._build_coref_head()

    @abstractmethod
    def _build_coref_head(self):
        pass


# TODO: span size features
# TODO: distance features
# TODO: s(i, eps) = 0
class BaseBertForCoreferenceResolution(BaseModeCoreferenceResolution, BaseModelBert):
    """
    mentions уже известны

    реализованы идеи следующих статей:
    https://arxiv.org/abs/1805.04893 - biaffine attention
    https://arxiv.org/abs/1804.05392 - hoi

    """
    def __init__(self, sess: tf.Session = None, config: Dict = None):
        """
        coref: {
            "use_birnn": False,
            "rnn": {...},
            "use_attn": True,
            "attn": {
                "hidden_dim": 128,
                "dropout": 0.3,
                "activation": "relu"
            }
            "hoi": {
                "order": 1,  # no hoi
                "w_dropout": 0.5,
                "w_dropout_policy": 0  # 0 - one mask; 1 - different mask
            },
            "biaffine": {
                ...
                "use_dep_prior": False
            }
        }

        :param sess:
        :param config:
        """
        super().__init__(sess=sess, config=config)

    def _build_coref_head(self):
        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

        if self.config["model"]["coref"]["use_birnn"]:
            self.birnn_re = StackedBiRNN(**self.config["model"]["coref"]["rnn"])
            emb_dim = self.config["model"]["coref"]["rnn"]["cell_dim"] * 2
        else:
            emb_dim = self.config["model"]["bert"]["dim"]

        self.entity_pairs_enc = GraphEncoder(**self.config["model"]["coref"]["biaffine"])
        multiple = 2 + int(self.config["model"]["coref"]["use_attn"])
        self.root_emb = tf.get_variable("root_emb", shape=[1, emb_dim * multiple], dtype=tf.float32)

        if self.config["model"]["coref"]["hoi"]["order"] > 1:
            self.w = tf.get_variable("w_update", shape=[emb_dim * multiple * 2, emb_dim * multiple], dtype=tf.float32)
            self.w_dropout = tf.keras.layers.Dropout(self.config["model"]["coref"]["hoi"]["w_dropout"])

        if self.config["model"]["coref"]["use_attn"]:
            self.dense_attn_1 = MLP(num_layers=1, **self.config["model"]["coref"]["attn"])
            self.dense_attn_2 = MLP(num_layers=1, hidden_dim=1, activation=None, dropout=None)

        # [batch_size, num_entities, num_entities + 1], [batch_size]
        self.logits_train, self.num_entities = self._build_coref_head_fn(bert_out=self.bert_out_train)
        self.logits_inference, _ = self._build_coref_head_fn(bert_out=self.bert_out_pred)

        # argmax
        self.labels_pred = tf.argmax(self.logits_inference, axis=-1)  # [batch_size, num_entities]

    def _build_coref_head_fn(self, bert_out):
        x, num_entities = self._get_entities_representation(bert_out=bert_out)

        batch_size = tf.shape(x)[0]
        x_root = tf.tile(self.root_emb, [batch_size, 1])
        x_root = x_root[:, None, :]

        num_entities_inner = num_entities + tf.ones_like(num_entities)

        # mask padding
        mask_pad = tf.sequence_mask(num_entities_inner)  # [batch_size, num_entities + 1]

        # mask antecedent
        n = tf.reduce_max(num_entities)
        mask_ant = tf.linalg.band_part(tf.ones((n, n + 1), dtype=tf.bool), -1, 0)  # lower-triangular

        mask = tf.logical_and(mask_pad[:, None, :], mask_ant[None, :, :])
        mask_additive = get_additive_mask(mask)

        def get_logits(enc, g):
            g_dep = tf.concat([x_root, g], axis=1)  # [batch_size, num_entities + 1, bert_dim]

            # encoding of pairs
            inputs = GraphEncoderInputs(head=g, dep=g_dep)
            logits = enc(inputs=inputs, training=self.training_ph)  # [N, num_ent, num_ent + 1, 1]

            # squeeze
            logits = tf.squeeze(logits, axis=[-1])  # [batch_size, num_entities, num_entities + 1]

            # mask
            logits += mask_additive

            return g_dep, logits

        # n = 1 - baseline
        # n = 2 - like in paper
        order = self.config["model"]["coref"]["hoi"]["order"]

        # 0 - one mask for each iteration
        # 1 - different mask on each iteration
        dropout_policy = self.config["model"]["coref"]["hoi"]["w_dropout_policy"]

        if dropout_policy == 0:
            w = self.w_dropout(self.w, training=self.training_ph)
        elif dropout_policy == 1:
            w = self.w
        else:
            raise NotImplementedError

        for i in range(order - 1):
            x_dep, logits = get_logits(self.entity_pairs_enc, x)

            # expected antecedent representation
            prob = tf.nn.softmax(logits, axis=-1)  # [batch_size, num_entities, num_entities + 1]
            a = tf.matmul(prob, x_dep)  # [batch_size, num_entities, bert_dim]

            # update
            if dropout_policy == 1:
                w = self.w_dropout(self.w, training=self.training_ph)
            f = tf.nn.sigmoid(tf.matmul(tf.concat([x, a], axis=-1), w))
            x = f * x + (1.0 - f) * a

        _, logits = get_logits(self.entity_pairs_enc, x)

        return logits, num_entities

    def _get_entities_representation(self, bert_out: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """


        bert_out - [batch_size, num_pieces, bert_dim]
        ner_labels - [batch_size, num_tokens, num_tokens]

        logits - [batch_size, num_entities_max, bert_bim or cell_dim * 2]
        num_entities - [batch_size]
        """
        # dropout
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        # birnn
        if self.birnn_re is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_re(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]
        #     d_model = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
        # else:
        #     d_model = self.config["model"]["bert"]["dim"]

        # ner labels
        num_tokens = tf.reduce_max(self.num_tokens_ph)
        batch_size = tf.shape(self.num_tokens_ph)[0]
        shape = tf.constant([batch_size, num_tokens, num_tokens], dtype=tf.int32)
        no_mention_id = 0
        ones = tf.ones_like(self.mention_coords_ph[:, :1])
        indices = tf.concat([self.mention_coords_ph, ones], axis=1)
        mention_coords_dense = get_dense_labels_from_indices(indices=indices, shape=shape, no_label_id=no_mention_id)

        # маскирование
        mask = upper_triangular(num_tokens, dtype=tf.int32)
        mention_coords_dense *= mask[None, :, :]

        # векторизация сущностей
        span_mask = tf.not_equal(mention_coords_dense, no_mention_id)  # [batch_size, num_tokens, num_tokens]
        start_coords, end_coords, num_entities = get_padded_coords_3d(mask_3d=span_mask)
        x_start = tf.gather_nd(x, start_coords)  # [N, num_entities, D]
        x_end = tf.gather_nd(x, end_coords)  # [N, num_entities, D]

        # attn
        grid, sequence_mask_span = get_span_indices(
            start_ids=start_coords[:, :, 1],
            end_ids=end_coords[:, :, 1]
        )  # ([batch_size, num_entities, span_size], [batch_size, num_entities, span_size])

        batch_size = tf.shape(x)[0]
        x_coord = tf.range(batch_size)[:, None, None, None]  # [batch_size, 1, 1, 1]
        grid_shape = tf.shape(grid)  # [3]
        x_coord = tf.tile(x_coord, [1, grid_shape[1], grid_shape[2], 1])  # [batch_size, num_entities, span_size, 1]
        y_coord = tf.expand_dims(grid, -1)  # [batch_size, num_entities, span_size, 1]
        coords = tf.concat([x_coord, y_coord], axis=-1)  # [batch_size, num_entities, span_size, 2]
        x_span = tf.gather_nd(x, coords)  # [batch_size, num_entities, span_size, d_model]
        # print(x_span)
        w = self.dense_attn_1(x_span)  # [batch_size, num_entities, span_size, H]
        w = self.dense_attn_2(w)  # [batch_size, num_entities, span_size, 1]
        sequence_mask_span = tf.expand_dims(sequence_mask_span, -1)
        w += get_additive_mask(sequence_mask_span)  # [batch_size, num_entities, span_size, 1]
        w = tf.nn.softmax(w, axis=2)  # [batch_size, num_entities, span_size, 1]
        x_span = tf.reduce_sum(x_span * w, axis=2)  # [batch_size, num_entities, d_model]

        # concat
        x_entity = tf.concat([x_start, x_end, x_span], axis=-1)  # [batch_size, num_entities, d_model * 3]

        return x_entity, num_entities

    def _set_placeholders(self):
        super()._set_placeholders()
        # [id_example, id_anaphora, id_antecedent]
        self.mention_coords_ph = tf.placeholder(tf.int32, shape=[None, 3], name="mention_coords")

    # TODO: мб вынести в BaseBertForCoreferenceResolution
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        pass

    # TODO: мб вынести в BaseBertForCoreferenceResolution
    def predict(self, examples: List[Example], **kwargs) -> None:
        pass


class BertForCoreferenceResolutionMentionPair(BaseBertForCoreferenceResolution):
    def __init__(self, sess: tf.Session = None, config: Dict = None):
        super().__init__(sess=sess, config=config)

    def _set_loss(self, *args, **kwargs):
        logits_shape = tf.shape(self.logits_train)
        labels = tf.scatter_nd(
            indices=self.mention_coords_ph[:, :-1], updates=self.mention_coords_ph[:, -1], shape=logits_shape[:2]
        )  # [batch_size, num_entities]
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.logits_train
        )  # [batch_size, num_entities]

        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)

        masked_per_example_loss = per_example_loss * sequence_mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(sequence_mask), tf.float32)
        num_pairs = tf.maximum(num_pairs, 1.0)
        self.loss = total_loss / num_pairs

    def _get_feed_dict(self, examples: List[Example], mode: str):
        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []

        # re
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            input_ids_i = []
            input_mask_i = []
            segment_ids_i = []
            first_pieces_coords_i = []

            # [CLS]
            input_ids_i.append(self.config["model"]["bert"]["cls_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            ptr = 1

            # tokens
            for t in x.tokens:
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.pieces)
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                ptr += num_pieces_ij

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            # coref labels
            if mode != ModeKeys.TEST:
                id2entity = {entity.id: entity for entity in x.entities}
                head2dep = {arc.head: id2entity[arc.dep] for arc in x.arcs}

                for entity in x.entities:
                    if entity.id in head2dep.keys():
                        dep_index = head2dep[entity.id].index + 1
                    else:
                        dep_index = 0
                    re_labels.append((i, entity.index, dep_index))

            # write
            num_pieces.append(len(input_ids_i))
            num_tokens.append(len(x.tokens))
            input_ids.append(input_ids_i)
            input_mask.append(input_mask_i)
            segment_ids.append(segment_ids_i)
            first_pieces_coords.append(first_pieces_coords_i)

        # padding
        pad_token_id = self.config["model"]["bert"]["pad_token_id"]
        num_tokens_max = max(num_tokens)
        num_pieces_max = max(num_pieces)
        for i in range(len(examples)):
            input_ids[i] += [pad_token_id] * (num_pieces_max - num_pieces[i])
            input_mask[i] += [0] * (num_pieces_max - num_pieces[i])
            segment_ids[i] += [0] * (num_pieces_max - num_pieces[i])
            first_pieces_coords[i] += [(i, 0)] * (num_tokens_max - num_tokens[i])

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0))

        training = mode == ModeKeys.TRAIN

        d = {
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,
            self.first_pieces_coords_ph: first_pieces_coords,
            self.num_pieces_ph: num_pieces,
            self.num_tokens_ph: num_tokens,
            self.training_ph: training
        }

        if mode != ModeKeys.TEST:
            d[self.mention_coords_ph] = re_labels

        return d


class BertForCoreferenceResolutionMentionRanking(BaseBertForCoreferenceResolution):
    def __init__(self, sess: tf.Session = None, config: Dict = None):
        super().__init__(sess=sess, config=config)

    def _set_loss(self):
        logits_shape = tf.shape(self.logits_train)
        updates = tf.ones_like(self.mention_coords_ph[:, 0])
        labels = tf.scatter_nd(
            indices=self.mention_coords_ph, updates=updates, shape=logits_shape
        )  # [batch_size, num_entities, num_entities + 1]

        # предполагается, что логиты уже маскированы по последнему измерению (pad, look-ahead)
        scores_model = tf.reduce_logsumexp(self.logits_train, axis=-1)  # [batch_size, num_entities]
        logits_gold = self.logits_train + get_additive_mask(labels)  # [batch_size, num_entities, num_entities + 1]
        scores_gold = tf.reduce_logsumexp(logits_gold, axis=-1)  # [batch_size, num_entities]
        per_example_loss = scores_model - scores_gold  # [batch_size, num_entities]

        # mask
        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)
        masked_per_example_loss = per_example_loss * sequence_mask

        # aggregate
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_entities_total = tf.cast(tf.reduce_sum(self.num_entities), tf.float32)
        num_entities_total = tf.maximum(num_entities_total, 1.0)
        self.loss = total_loss / num_entities_total

    def _get_feed_dict(self, examples: List[Example], mode: str):
        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []

        # re
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            input_ids_i = []
            input_mask_i = []
            segment_ids_i = []
            first_pieces_coords_i = []

            # [CLS]
            input_ids_i.append(self.config["model"]["bert"]["cls_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            ptr = 1

            # tokens
            for t in x.tokens:
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.pieces)
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                ptr += num_pieces_ij

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            # ner, re
            if mode != ModeKeys.TEST:
                # id2entity = {entity.id: entity for entity in x.entities}
                id2entity = {}
                chain2entities = defaultdict(set)

                for entity in x.entities:
                    assert isinstance(entity.index, int)
                    assert isinstance(entity.id_chain, int)
                    id2entity[entity.id] = entity
                    chain2entities[entity.id_chain].add(entity)

                for entity in x.entities:
                    antecedents = []
                    for entity_chain in chain2entities[entity.id_chain]:
                        if entity_chain.index < entity.index:
                            antecedents.append(entity_chain.index)
                    if len(antecedents) > 0:
                        for id_dep in antecedents:
                            re_labels.append((i, entity.index, id_dep + 1))
                    else:
                        re_labels.append((i, entity.index, 0))

            # write
            num_pieces.append(len(input_ids_i))
            num_tokens.append(len(x.tokens))
            input_ids.append(input_ids_i)
            input_mask.append(input_mask_i)
            segment_ids.append(segment_ids_i)
            first_pieces_coords.append(first_pieces_coords_i)

        # padding
        pad_token_id = self.config["model"]["bert"]["pad_token_id"]
        num_tokens_max = max(num_tokens)
        num_pieces_max = max(num_pieces)
        for i in range(len(examples)):
            input_ids[i] += [pad_token_id] * (num_pieces_max - num_pieces[i])
            input_mask[i] += [0] * (num_pieces_max - num_pieces[i])
            segment_ids[i] += [0] * (num_pieces_max - num_pieces[i])
            first_pieces_coords[i] += [(i, 0)] * (num_tokens_max - num_tokens[i])

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0))

        training = mode == ModeKeys.TRAIN

        d = {
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,
            self.first_pieces_coords_ph: first_pieces_coords,
            self.num_pieces_ph: num_pieces,
            self.num_tokens_ph: num_tokens,
            self.training_ph: training
        }

        if mode != ModeKeys.TEST:
            d[self.mention_coords_ph] = re_labels

        return d
