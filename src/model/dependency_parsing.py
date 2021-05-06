from abc import abstractmethod
from typing import Dict, List

import tensorflow as tf
import numpy as np

from src.model.base import BaseModel, BaseModelBert, ModeKeys
from src.model.layers import StackedBiRNN, GraphEncoder, GraphEncoderInputs
from src.model.utils import get_additive_mask
from src.data.base import Example
from src.utils import batches_gen, mst


class BaseModeDependencyParsing(BaseModel):
    dep_scope = "dependency_parser"

    def __init__(self, sess: tf.Session = None, config: Dict = None, rel_enc: Dict = None):
        super().__init__(sess=sess, config=config)

        # PLACEHOLDERS
        self.labels_ph = None  # [id_example, id_head, id_dep, id_rel]

        # LAYERS
        self.birnn = None
        self.arc_enc = None
        self.type_enc = None

        # TENSORS
        self.logits_arc_train = None
        self.logits_type_train = None
        self.s_arc = None
        self.type_labels_pred = None
        # for debug:
        self.total_loss_arc = None
        self.total_loss_type = None

        self._rel_enc = None
        self._inv_rel_enc = None

        self.rel_enc = rel_enc

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.dep_scope):
            self._build_dependency_parser()

    @abstractmethod
    def _build_dependency_parser(self):
        pass

    @property
    def rel_enc(self):
        return self._rel_enc

    @property
    def inv_rel_enc(self):
        return self._inv_rel_enc

    @rel_enc.setter
    def rel_enc(self, rel_enc: Dict):
        self._rel_enc = rel_enc
        if rel_enc is not None:
            self._inv_rel_enc_enc = {v: k for k, v in rel_enc.items()}


class BertForDependencyParsing(BaseModeDependencyParsing, BaseModelBert):
    def __init__(self, sess: tf.Session = None, config: Dict = None, rel_enc: Dict = None):
        super().__init__(sess=sess, config=config, rel_enc=rel_enc)

    def _build_dependency_parser(self):
        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

        if self.config["model"]["parser"]["use_birnn"]:
            self.birnn = StackedBiRNN(**self.config["model"]["parser"]["rnn"])

        with tf.variable_scope("arc_encoder"):
            self.arc_enc = GraphEncoder(**self.config["model"]["parser"]["biaffine_arc"])
        with tf.variable_scope("type_encoder"):
            self.type_enc = GraphEncoder(**self.config["model"]["parser"]["biaffine_type"])

        self.logits_arc_train, self.logits_type_train = self._build_dependency_parser_fn(bert_out=self.bert_out_train)
        logits_arc_pred, logits_type_pred = self._build_dependency_parser_fn(bert_out=self.bert_out_pred)

        self.s_arc = tf.nn.softmax(logits_arc_pred, axis=-1)  # [N, T, T + 1]
        self.type_labels_pred = tf.argmax(logits_type_pred, axis=-1)  # [N, T, T + 1]

    def _build_dependency_parser_fn(self, bert_out: tf.Tensor):
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [N, T + 1, bert_dim]

        # birnn
        if self.birnn is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph + 1)
            x = self.birnn(x, training=self.training_ph, mask=sequence_mask)  # [N, T + 1, cell_dim * 2]

        x_wo_root = x[:, 1:, :]  # [N, T, D]

        # arc
        enc_inputs = GraphEncoderInputs(head=x_wo_root, dep=x)
        logits_arc = self.arc_enc(enc_inputs, training=self.training_ph)  # [N, T, T + 1, 1]
        logits_arc = tf.squeeze(logits_arc, axis=-1)  # [N, T, T + 1]

        # type
        enc_inputs = GraphEncoderInputs(head=x_wo_root, dep=x)
        logits_type = self.type_enc(enc_inputs, training=self.training_ph)  # [N, T, T + 1, num_relations]

        # mask (last dimention only due to softmax)
        mask = tf.sequence_mask(self.num_tokens_ph + 1, dtype=tf.float32)  # [N, T + 1]
        logits_arc += get_additive_mask(mask[:, None, :])  # [N, T, T + 1]
        return logits_arc, logits_type

    def _set_placeholders(self):
        super()._set_placeholders()
        self.labels_ph = tf.placeholder(tf.int32, shape=[None, 4], name="labels")

    def _set_loss(self, *args, **kwargs):
        total_num_tokens = tf.reduce_sum(self.num_tokens_ph)

        # arc
        labels_arc = tf.scatter_nd(
            indices=self.labels_ph[:, :2], updates=self.labels_ph[:, 2], shape=tf.shape(self.logits_arc_train)[:2]
        )  # [N, T]
        per_example_loss_arc = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_arc, logits=self.logits_arc_train
        )  # [N, T]
        sequence_mask = tf.sequence_mask(self.num_tokens_ph, dtype=tf.float32)  # [N, T]
        masked_per_example_loss_arc = per_example_loss_arc * sequence_mask
        total_loss_arc = tf.reduce_sum(masked_per_example_loss_arc)
        loss_arc = total_loss_arc / tf.cast(total_num_tokens, tf.float32)

        # type
        labels_type = tf.scatter_nd(
            indices=self.labels_ph[:, :3], updates=self.labels_ph[:, 3], shape=tf.shape(self.logits_type_train)[:3]
        )  # [N, T, T + 1]
        per_example_loss_type = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_type, logits=self.logits_type_train
        )  # [N, T, T + 1]
        masked_per_example_loss_type = tf.gather_nd(per_example_loss_type, self.labels_ph[:, :3])
        total_loss_type = tf.reduce_sum(masked_per_example_loss_type)
        # num_edges = num_tokens - 1 (tree cond) + 1 (fake root node) = num_tokens
        loss_type = total_loss_type / tf.cast(total_num_tokens, tf.float32)

        self.loss = loss_arc + loss_type

        # for debug
        self.total_loss_arc = total_loss_arc
        self.total_loss_type = total_loss_type

    def _get_feed_dict(self, examples: List[Example], mode: str):
        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []

        labels = []

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

            # [ROOT]
            input_ids_i.append(self.config["model"]["bert"]["root_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            first_pieces_coords_i.append((i, 1))

            ptr = 2

            # tokens
            for j, t in enumerate(x.tokens):
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.pieces)
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                ptr += num_pieces_ij

                if mode != ModeKeys.TEST:
                    assert isinstance(t.id_head, int)
                    assert isinstance(t.rel, str)
                    if t.id_head == -1:  # TODO: учесть специальный индекс для root при парсинге примеров
                        k = 0
                    else:
                        k = t.id_head + 1
                    labels.append((i, j, k, self.rel_enc[t.rel]))

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

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

            if len(labels) == 0:
                labels.append((0, 0, 0, 0))

            d[self.labels_ph] = labels

        return d

    # TODO: implement
    def predict(self, examples: List[Example], **kwargs) -> None:
        pass

    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        """chunks always sentence-level"""
        chunks = []
        for x in examples:
            assert len(x.chunks) > 0
            chunks += x.chunks

        num_tokens_total = 0
        num_heads_correct = 0
        num_heads_labels_correct = 0

        total_loss_arc = 0.0
        total_loss_type = 0.0

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            s_arc, type_labels_pred, loss_arc_i, loss_type_i = self.sess.run(
                [self.s_arc, self.type_labels_pred, self.total_loss_arc, self.total_loss_type],
                feed_dict=feed_dict
            )
            total_loss_arc += loss_arc_i
            total_loss_type += loss_type_i

            for i, x in enumerate(batch):
                num_tokens_i = len(x.tokens)
                s_arc_i = s_arc[i, :num_tokens_i, :num_tokens_i + 1]  # [T, T + 1]
                root_scores = np.zeros_like(s_arc_i[:1, :])
                root_scores[0] = 1.0
                s_arc_i = np.concatenate([root_scores, s_arc_i], axis=0)  # [T + 1, T + 1]
                head_ids = mst(s_arc_i)  # [T + 1]; head_ids[0] = 0, heads[1:] in range [0, num_tokens_i]

                for j, t in enumerate(x.tokens):
                    num_tokens_total += 1
                    head_pred = head_ids[j + 1]
                    if head_pred == t.id_head + 1:
                        num_heads_correct += 1
                        id_label_pred = type_labels_pred[i, j, head_pred]
                        if id_label_pred == self.rel_enc[t.rel]:
                            num_heads_labels_correct += 1

        # loss
        loss_arc = total_loss_arc / num_tokens_total
        loss_type = total_loss_type / num_tokens_total
        loss = loss_arc + loss_type

        # metrics
        uas = num_heads_correct / num_tokens_total
        las = num_heads_labels_correct / num_tokens_total

        performance_info = {
            "loss": loss,
            "loss_arc": loss_arc,
            "loss_type": loss_type,
            "score": las,
            "uas": uas,
            "las": las
        }

        return performance_info
