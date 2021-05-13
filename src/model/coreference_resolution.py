import copy
from typing import Dict, List
from collections import defaultdict

import tensorflow as tf
import numpy as np

from src.data.base import Example, Arc
from src.data.io import to_conll
from src.model.base import BaseModeCoreferenceResolution, BaseModelBert, ModeKeys
from src.model.layers import GraphEncoder, GraphEncoderInputs, MLP
from src.model.utils import (
    get_additive_mask,
    get_entities_representation,
    get_sent_pairs_to_predict_for,
    get_sent_ids_to_predict_for
)
from src.metrics import get_coreferense_resolution_metrics
from src.utils import batches_gen, get_connected_components, parse_conll_metrics, log


__all__ = ["BertForCoreferenceResolutionMentionPair", "BertForCoreferenceResolutionMentionRanking"]


# TODO: span size features
# TODO: distance features
# TODO: s(i, eps) = 0
# TODO: ченкуть реализацию инференса здесь:
#  https://github.com/kentonl/e2e-coref/blob/9d1ee1972f6e34eb5d1dcbb1fd9b9efdf53fc298/coref_model.py#L498
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

        self.birnn = None
        self.w = None
        self.w_dropout = None
        self.ff_attn = None

    def _build_coref_head(self):
        x_ent_train, self.num_entities = self._get_entities_representation(bert_out=self.bert_out_train)
        self.logits_train = self._get_entity_pairs_logits(x_ent_train)
        x_ent_pred, _ = self._get_entities_representation(bert_out=self.bert_out_pred)
        self.logits_inference = self._get_entity_pairs_logits(x_ent_pred)
        self.labels_pred = tf.argmax(self.logits_inference, axis=-1)  # [batch_size, num_entities]

    def _set_layers(self):
        super()._set_layers()

        if self.config["model"]["coref"]["use_birnn"]:
            emb_dim = self.config["model"]["coref"]["rnn"]["cell_dim"] * 2
        else:
            emb_dim = self.config["model"]["bert"]["params"]["hidden_size"]

        self.entity_pairs_enc = GraphEncoder(**self.config["model"]["coref"]["biaffine"])
        multiple = 2 + int(self.config["model"]["coref"]["use_attn"])
        self.root_emb = tf.get_variable("root_emb", shape=[1, emb_dim * multiple], dtype=tf.float32)

        if self.config["model"]["coref"]["hoi"]["order"] > 1:
            self.w = tf.get_variable("w_update", shape=[emb_dim * multiple * 2, emb_dim * multiple], dtype=tf.float32)
            self.w_dropout = tf.keras.layers.Dropout(self.config["model"]["coref"]["hoi"]["w_dropout"])

        if self.config["model"]["coref"]["use_attn"]:
            self.ff_attn = MLP(
                num_layers=2,
                hidden_dim=[self.config["model"]["coref"]["attn"]["hidden_dim"], 1],
                activation=[self.config["model"]["coref"]["attn"]["activation"], None],
                dropout=[self.config["model"]["coref"]["attn"]["dropout"], None]
            )

    def _get_entities_representation(self, bert_out: tf.Tensor):
        x_token = self._get_token_level_embeddings(bert_out=bert_out)
        updates = tf.ones_like(self.mention_spans_ph[:, :1])
        ner_labels = tf.concat([self.mention_spans_ph, updates], axis=1)
        x_entity, num_entities = get_entities_representation(
            x=x_token, ner_labels=ner_labels, sparse_labels=True, ff_attn=self.ff_attn
        )
        return x_entity, num_entities

    def _get_entity_pairs_logits(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        x_root = tf.tile(self.root_emb, [batch_size, 1])
        x_root = x_root[:, None, :]

        num_entities_inner = self.num_entities + tf.ones_like(self.num_entities)

        # mask padding
        mask_pad = tf.sequence_mask(num_entities_inner)  # [batch_size, num_entities + 1]

        # mask antecedent
        n = tf.reduce_max(self.num_entities)
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

        if order > 1:
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

        return logits

    def _set_placeholders(self):
        super()._set_placeholders()
        self.mention_spans_ph = tf.placeholder(tf.int32, shape=[None, 3], name="mention_spans_ph")
        self.labels_ph = tf.placeholder(tf.int32, shape=[None, 3], name="labels_ph")

    @log
    def predict(self, examples: List[Example], flat_chains: bool = True, **kwargs) -> None:
        # TODO: как-то обработать случай отсутствия сущнсоетй

        # проверка примеров
        chunks = []
        id_to_num_sentences = {}
        for x in examples:
            assert len(x.arcs) == 0
            # assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for chunk in x.chunks:
                assert chunk.parent is not None, f"[{x.id}] parent for chunk {chunk.id} is not set. " \
                    f"It is not a problem, but must be set for clarity"
                chunks.append(chunk)
            id_to_num_sentences[x.id] = x.tokens[-1].id_sent + 1

        assert len(id_to_num_sentences) == len(examples), f"examples must have unique ids, " \
            f"but got {len(id_to_num_sentences)} unique ids among {len(examples)} examples"

        head2dep = {}  # (file, head) -> {dep, score}
        window = self.config["inference"]["window"]

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.TEST)
            re_labels_pred, re_logits_pred = self.sess.run(
                [self.labels_pred, self.logits_inference],
                feed_dict=feed_dict
            )
            # re_labels_pred: np.ndarray, shape [batch_size, num_entities], dtype np.int32
            # values in range [0, num_ent]; 0 means no dep.
            # re_logits_pred: np.ndarray, shape [batch_size, num_entities, num_entities + 1], dtype np.float32

            for i in range(len(batch)):
                chunk = batch[i]

                num_entities_chunk = len(chunk.entities)
                index2entity = {entity.index: entity for entity in chunk.entities}
                assert len(index2entity) == num_entities_chunk

                num_sentences = id_to_num_sentences[chunk.parent]
                end_rel = chunk.tokens[-1].id_sent - chunk.tokens[0].id_sent
                assert end_rel < window, f"[{chunk.id}] relative end {end_rel} >= window size {window}"
                is_first = chunk.tokens[0].id_sent == 0
                is_last = chunk.tokens[-1].id_sent == num_sentences - 1
                pairs = get_sent_pairs_to_predict_for(end=end_rel, is_first=is_first, is_last=is_last, window=window)

                # предсказанные лейблы, которые можно получить из предиктов для кусочка chunk
                for id_sent_rel_a, id_sent_rel_b in pairs:
                    id_sent_abs_a = id_sent_rel_a + chunk.tokens[0].id_sent
                    id_sent_abs_b = id_sent_rel_b + chunk.tokens[0].id_sent
                    # for idx_head, idx_dep in enumerate(re_labels_pred_i):
                    for idx_head in range(num_entities_chunk):
                        idx_dep = re_labels_pred[i, idx_head]
                        # нет исходящего ребра или находится дальше по тексту
                        if idx_dep == 0 or idx_dep >= idx_head + 1:
                            continue
                        score = re_logits_pred[i, idx_head, idx_dep]
                        # это условие акутально только тогда,
                        # когда реализована оригинальная логика: s(i, eps) = 0
                        # if score <= 0.0:
                        #     continue
                        head = index2entity[idx_head]
                        dep = index2entity[idx_dep - 1]
                        id_sent_head = head.tokens[0].id_sent
                        id_sent_dep = dep.tokens[0].id_sent
                        if (id_sent_head == id_sent_abs_a and id_sent_dep == id_sent_abs_b) or \
                                (id_sent_head == id_sent_abs_b and id_sent_dep == id_sent_abs_a):
                            key_head = chunk.parent, head.id
                            if key_head in head2dep:
                                if head2dep[key_head]["score"] < score:
                                    head2dep[key_head] = {"dep": dep.id, "score": score}
                                else:
                                    pass
                            else:
                                head2dep[key_head] = {"dep": dep.id, "score": score}

        # присвоение id_chain
        for x in examples:
            id2entity = {}
            g = {}
            for entity in x.entities:
                g[entity.id] = set()
                id2entity[entity.id] = entity
            for entity in x.entities:
                key = x.id, entity.id
                if key in head2dep:
                    id_dep = head2dep[key]["dep"]
                    g[entity.id].add(id_dep)
                    if not flat_chains:
                        id_arc = "R" + str(len(x.arcs))
                        arc = Arc(id=id_arc, head=entity.id, dep=id_dep, rel=self.coref_rel)
                        x.arcs.append(arc)

            # print(g)
            components = get_connected_components(g)

            for id_chain, comp in enumerate(components):
                entities_comp = []
                for id_entity in comp:
                    entity = id2entity[id_entity]
                    entity.id_chain = id_chain
                    entities_comp.append(entity)
                if flat_chains and len(comp) > 1:
                    entities_comp_sorted = sorted(entities_comp, key=lambda e: (e.tokens[0].index_abs, e.tokens[-1].index_abs))
                    for i in range(len(comp) - 1):
                        dep = entities_comp_sorted[i]
                        head = entities_comp_sorted[i + 1]
                        id_arc = "R" + str(len(x.arcs))
                        arc = Arc(id=id_arc, head=head.id, dep=dep.id, rel=self.coref_rel)
                        x.arcs.append(arc)


class BertForCoreferenceResolutionMentionPair(BaseBertForCoreferenceResolution):
    def __init__(self, sess: tf.Session = None, config: Dict = None):
        super().__init__(sess=sess, config=config)

    def _set_loss(self, *args, **kwargs):
        logits_shape = tf.shape(self.logits_train)
        labels = tf.scatter_nd(
            indices=self.labels_ph[:, :-1], updates=self.labels_ph[:, -1], shape=logits_shape[:2]
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
        self.total_loss = total_loss
        self.loss_denominator = num_pairs

    def _get_feed_dict(self, examples: List[Example], mode: str):
        bert_inputs = self._get_bert_input_for_feed_dict(examples)

        mention_spans = []
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            # mention spans
            for entity in x.entities:
                assert entity.label is not None
                start = entity.tokens[0].index_rel
                assert start is not None
                end = entity.tokens[-1].index_rel
                assert end is not None
                mention_spans.append((i, start, end))

            # coref labels
            if mode != ModeKeys.TEST:
                id2entity = {entity.id: entity for entity in x.entities}
                head2dep = {arc.head: id2entity[arc.dep] for arc in x.arcs}

                for entity in x.entities:
                    assert isinstance(entity.index, int)
                    if entity.id in head2dep.keys():
                        dep_index = head2dep[entity.id].index + 1
                    else:
                        dep_index = 0
                    re_labels.append((i, entity.index, dep_index))

        if len(mention_spans) == 0:
            mention_spans.append((0, 0, 0))

        d = {
            self.input_ids_ph: bert_inputs.input_ids,
            self.input_mask_ph: bert_inputs.input_mask,
            self.segment_ids_ph: bert_inputs.segment_ids,
            self.first_pieces_coords_ph: bert_inputs.first_pieces_coords,
            self.mention_spans_ph: mention_spans,
            self.num_pieces_ph: bert_inputs.num_pieces,
            self.num_tokens_ph: bert_inputs.num_tokens,
            self.training_ph: mode == ModeKeys.TRAIN
        }

        if mode != ModeKeys.TEST:
            if len(re_labels) == 0:
                re_labels.append((0, 0, 0))
            d[self.labels_ph] = re_labels

        return d

    # TODO: много копипасты из predict
    @log
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        examples_valid_copy = copy.deepcopy(examples)

        chunks = []
        id_to_num_sentences = {}
        num_entities_total_example_level = 0
        num_entities_total_chunk_level = 0
        num_chains_true = 0
        total_loss = 0.0
        loss_denominator = 0
        num_right_preds = 0
        num_chains_pred = 0

        # проверка примеров
        for x in examples:
            # assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for chunk in x.chunks:
                assert chunk.parent is not None, f"[{x.id}] parent for chunk {chunk.id} is not set. " \
                    f"It is not a problem, but must be set for clarity"
                chunks.append(chunk)
            chain_ids = set()
            for entity in x.entities:
                assert entity.id_chain is not None, f"[{x.id}] entity {entity.id} has no id_chain"
                num_entities_total_example_level += 1
                chain_ids.add(entity.id_chain)
            id_to_num_sentences[x.id] = x.tokens[-1].id_sent + 1
            num_chains_true += len(chain_ids)

        assert len(id_to_num_sentences) == len(examples), f"examples must have unique ids, " \
            f"but got {len(id_to_num_sentences)} unique ids among {len(examples)} examples"

        head2dep = {}  # (file, head) -> {dep, score}
        window = self.config["inference"]["window"]

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )

        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            total_loss_i, d, re_labels_pred, re_logits_pred = self.sess.run(
                [self.total_loss, self.loss_denominator, self.labels_pred, self.logits_inference],
                feed_dict=feed_dict
            )
            total_loss += total_loss_i
            loss_denominator += d
            # re_labels_pred: np.ndarray, shape [batch_size, num_entities], dtype np.int32
            # values in range [0, num_ent]; 0 means no dep.
            # re_logits_pred: np.ndarray, shape [batch_size, num_entities, num_entities + 1], dtype np.float32

            for i in range(len(batch)):
                chunk = batch[i]

                num_entities_chunk = len(chunk.entities)
                num_entities_total_chunk_level += num_entities_chunk
                index2entity = {}
                entity2index = {}
                for entity in chunk.entities:
                    index2entity[entity.index] = entity
                    entity2index[entity.id] = entity.index
                assert len(index2entity) == num_entities_chunk
                assert len(entity2index) == num_entities_chunk

                num_sentences = id_to_num_sentences[chunk.parent]
                end_rel = chunk.tokens[-1].id_sent - chunk.tokens[0].id_sent
                assert end_rel < window, f"[{chunk.id}] relative end {end_rel} >= window size {window}"
                is_first = chunk.tokens[0].id_sent == 0
                is_last = chunk.tokens[-1].id_sent == num_sentences - 1
                pairs = get_sent_pairs_to_predict_for(end=end_rel, is_first=is_first, is_last=is_last, window=window)

                index_head_to_index_dep = {}
                for arc in chunk.arcs:
                    idx_head = entity2index[arc.head]
                    assert idx_head not in index_head_to_index_dep, \
                        f"[{chunk.id}] entity {arc.head} has more than one antecedent"
                    idx_dep = entity2index[arc.dep] + 1
                    index_head_to_index_dep[idx_head] = idx_dep

                for idx_head in range(num_entities_chunk):
                    idx_dep_pred = re_labels_pred[i, idx_head]
                    idx_dep_true = index_head_to_index_dep.get(idx_head, 0)
                    if idx_dep_true == idx_dep_pred:
                        num_right_preds += 1

                # предсказанные лейблы, которые можно получить из предиктов для кусочка chunk
                for id_sent_rel_a, id_sent_rel_b in pairs:
                    id_sent_abs_a = id_sent_rel_a + chunk.tokens[0].id_sent
                    id_sent_abs_b = id_sent_rel_b + chunk.tokens[0].id_sent
                    # for idx_head, idx_dep in enumerate(re_labels_pred_i):
                    for idx_head in range(num_entities_chunk):
                        idx_dep = re_labels_pred[i, idx_head]
                        # нет исходящего ребра или находится дальше по тексту
                        if idx_dep == 0 or idx_dep >= idx_head + 1:
                            continue
                        score = re_logits_pred[i, idx_head, idx_dep]
                        # это условие акутально только тогда,
                        # когда реализована оригинальная логика: s(i, eps) = 0
                        # if score <= 0.0:
                        #     continue
                        head = index2entity[idx_head]
                        dep = index2entity[idx_dep - 1]
                        id_sent_head = head.tokens[0].id_sent
                        id_sent_dep = dep.tokens[0].id_sent
                        if (id_sent_head == id_sent_abs_a and id_sent_dep == id_sent_abs_b) or \
                                (id_sent_head == id_sent_abs_b and id_sent_dep == id_sent_abs_a):
                            key_head = chunk.parent, head.id
                            if key_head in head2dep:
                                if head2dep[key_head]["score"] < score:
                                    head2dep[key_head] = {"dep": dep.id, "score": score}
                                else:
                                    pass
                            else:
                                head2dep[key_head] = {"dep": dep.id, "score": score}

        # присвоение id_chain
        for x in examples_valid_copy:
            id2entity = {}
            g = {}
            for entity in x.entities:
                g[entity.id] = set()
                id2entity[entity.id] = entity
                entity.id_chain = None
            for entity in x.entities:
                key = x.id, entity.id
                if key in head2dep:
                    dep = head2dep[key]["dep"]
                    g[entity.id].add(dep)

            # print(g)
            components = get_connected_components(g)

            for id_chain, comp in enumerate(components):
                num_chains_pred += 1
                for id_entity in comp:
                    id2entity[id_entity].id_chain = id_chain

        # compute performance info
        loss = total_loss / loss_denominator
        # print("loss:", loss)
        # print("total loss:", total_loss)
        # print("denominator:", loss_denominator)

        to_conll(examples=examples, path=self.config["valid"]["path_true"])
        to_conll(examples=examples_valid_copy, path=self.config["valid"]["path_pred"])

        metrics = {}
        for metric in ["muc", "bcub", "ceafm", "ceafe", "blanc"]:
            stdout = get_coreferense_resolution_metrics(
                path_true=self.config["valid"]["path_true"],
                path_pred=self.config["valid"]["path_pred"],
                scorer_path=self.config["valid"]["scorer_path"],
                metric=metric
            )
            is_blanc = metric == "blanc"
            metrics[metric] = parse_conll_metrics(stdout=stdout, is_blanc=is_blanc)

        score = (metrics["muc"]["f1"] + metrics["bcub"]["f1"] + metrics["ceafm"]["f1"] + metrics["ceafe"]["f1"]) / 4.0
        accuracy = num_right_preds / num_entities_total_chunk_level
        d = {
            "loss": loss,
            "score": score,
            "metrics": metrics,
            "antecedent_prediction_accuracy": accuracy,
            "num_entities": num_entities_total_example_level,
            "num_chains_true": num_chains_true,
            "num_chains_pred": num_chains_pred
        }
        return d


class BertForCoreferenceResolutionMentionRanking(BaseBertForCoreferenceResolution):
    def __init__(self, sess: tf.Session = None, config: Dict = None):
        super().__init__(sess=sess, config=config)

    def _set_loss(self):
        logits_shape = tf.shape(self.logits_train)
        updates = tf.ones_like(self.labels_ph[:, 0])
        labels = tf.scatter_nd(
            indices=self.labels_ph, updates=updates, shape=logits_shape
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
        self.total_loss = total_loss
        self.loss_denominator = num_entities_total

    def _get_feed_dict(self, examples: List[Example], mode: str):
        bert_inputs = self._get_bert_input_for_feed_dict(examples)

        mention_spans = []
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            # mention spans
            for entity in x.entities:
                assert entity.label is not None
                start = entity.tokens[0].index_rel
                assert start is not None
                end = entity.tokens[-1].index_rel
                assert end is not None
                mention_spans.append((i, start, end))

            # coref
            if mode != ModeKeys.TEST:
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

        if len(mention_spans) == 0:
            mention_spans.append((0, 0, 0))

        d = {
            self.input_ids_ph: bert_inputs.input_ids,
            self.input_mask_ph: bert_inputs.input_mask,
            self.segment_ids_ph: bert_inputs.segment_ids,
            self.first_pieces_coords_ph: bert_inputs.first_pieces_coords,
            self.mention_spans_ph: mention_spans,
            self.num_pieces_ph: bert_inputs.num_pieces,
            self.num_tokens_ph: bert_inputs.num_tokens,
            self.training_ph: mode == ModeKeys.TRAIN
        }

        if mode != ModeKeys.TEST:
            if len(re_labels) == 0:
                re_labels.append((0, 0, 0))
            d[self.labels_ph] = re_labels

        return d

    # TODO: много копипасты из predict
    # TODO: нет фильтрации по длине
    @log
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        examples_valid_copy = copy.deepcopy(examples)  # в случае смены примеров логика выше будет неверна

        chunks = []
        id_to_num_sentences = {}
        num_entities_total_example_level = 0
        num_chains_true = 0
        num_chains_pred = 0
        total_loss = 0.0
        loss_denominator = 0

        # проверка примеров
        for x in examples:
            # assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for chunk in x.chunks:
                assert chunk.parent is not None, f"[{x.id}] parent for chunk {chunk.id} is not set. " \
                    f"It is not a problem, but must be set for clarity"
                chunks.append(chunk)
            chain_ids = set()
            for entity in x.entities:
                assert entity.id_chain is not None, f"[{x.id}] entity {entity.id} has no id_chain"
                num_entities_total_example_level += 1
                chain_ids.add(entity.id_chain)
            id_to_num_sentences[x.id] = x.tokens[-1].id_sent + 1
            num_chains_true += len(chain_ids)

        assert len(id_to_num_sentences) == len(examples), f"examples must have unique ids, " \
            f"but got {len(id_to_num_sentences)} unique ids among {len(examples)} examples"

        head2dep = {}  # (file, head) -> {dep, score}
        window = self.config["inference"]["window"]

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            total_loss_i, d, re_labels_pred, re_logits_pred = self.sess.run(
                [self.total_loss, self.loss_denominator, self.labels_pred, self.logits_inference],
                feed_dict=feed_dict
            )
            total_loss += total_loss_i
            loss_denominator += d
            # re_labels_pred: np.ndarray, shape [batch_size, num_entities], dtype np.int32
            # values in range [0, num_ent]; 0 means no dep.
            # re_logits_pred: np.ndarray, shape [batch_size, num_entities, num_entities + 1], dtype np.float32

            for i in range(len(batch)):
                chunk = batch[i]

                num_entities_chunk = len(chunk.entities)
                index2entity = {entity.index: entity for entity in chunk.entities}
                assert len(index2entity) == num_entities_chunk

                num_sentences = id_to_num_sentences[chunk.parent]
                end_rel = chunk.tokens[-1].id_sent - chunk.tokens[0].id_sent
                assert end_rel < window, f"[{chunk.id}] relative end {end_rel} >= window size {window}"
                is_first = chunk.tokens[0].id_sent == 0
                is_last = chunk.tokens[-1].id_sent == num_sentences - 1
                pairs = get_sent_pairs_to_predict_for(end=end_rel, is_first=is_first, is_last=is_last, window=window)

                # предсказанные лейблы, которые можно получить из предиктов для кусочка chunk
                for id_sent_rel_a, id_sent_rel_b in pairs:
                    id_sent_abs_a = id_sent_rel_a + chunk.tokens[0].id_sent
                    id_sent_abs_b = id_sent_rel_b + chunk.tokens[0].id_sent
                    # for idx_head, idx_dep in enumerate(re_labels_pred_i):
                    for idx_head in range(num_entities_chunk):
                        idx_dep = re_labels_pred[i, idx_head]
                        # нет исходящего ребра или находится дальше по тексту
                        if idx_dep == 0 or idx_dep >= idx_head + 1:
                            continue
                        score = re_logits_pred[i, idx_head, idx_dep]
                        # это условие акутально только тогда,
                        # когда реализована оригинальная логика: s(i, eps) = 0
                        # if score <= 0.0:
                        #     continue
                        head = index2entity[idx_head]
                        dep = index2entity[idx_dep - 1]
                        id_sent_head = head.tokens[0].id_sent
                        id_sent_dep = dep.tokens[0].id_sent
                        if (id_sent_head == id_sent_abs_a and id_sent_dep == id_sent_abs_b) or \
                                (id_sent_head == id_sent_abs_b and id_sent_dep == id_sent_abs_a):
                            key_head = chunk.parent, head.id
                            if key_head in head2dep:
                                if head2dep[key_head]["score"] < score:
                                    head2dep[key_head] = {"dep": dep.id, "score": score}
                                else:
                                    pass
                            else:
                                head2dep[key_head] = {"dep": dep.id, "score": score}

        # присвоение id_chain
        for x in examples_valid_copy:
            id2entity = {}
            g = {}
            for entity in x.entities:
                g[entity.id] = set()
                id2entity[entity.id] = entity
                entity.id_chain = None
            for entity in x.entities:
                key = x.id, entity.id
                if key in head2dep:
                    dep = head2dep[key]["dep"]
                    g[entity.id].add(dep)

            # print(g)
            components = get_connected_components(g)

            for id_chain, comp in enumerate(components):
                num_chains_pred += 1
                for id_entity in comp:
                    id2entity[id_entity].id_chain = id_chain

        # compute performance info
        loss = total_loss / loss_denominator

        to_conll(examples=examples, path=self.config["valid"]["path_true"])
        to_conll(examples=examples_valid_copy, path=self.config["valid"]["path_pred"])

        metrics = {}
        for metric in ["muc", "bcub", "ceafm", "ceafe", "blanc"]:
            stdout = get_coreferense_resolution_metrics(
                path_true=self.config["valid"]["path_true"],
                path_pred=self.config["valid"]["path_pred"],
                scorer_path=self.config["valid"]["scorer_path"],
                metric=metric
            )
            is_blanc = metric == "blanc"
            metrics[metric] = parse_conll_metrics(stdout=stdout, is_blanc=is_blanc)

        score = (metrics["muc"]["f1"] + metrics["bcub"]["f1"] + metrics["ceafm"]["f1"] + metrics["ceafe"]["f1"]) / 4.0
        d = {
            "loss": loss,
            "score": score,
            "metrics": metrics,
            "num_entities": num_entities_total_example_level,
            "num_chains_true": num_chains_true,
            "num_chains_pred": num_chains_pred
        }
        return d


class BertForCoreferenceResolutionMentionRankingNewInference(BertForCoreferenceResolutionMentionRanking):
    def __init__(self, sess: tf.Session = None, config: Dict = None):
        super().__init__(sess=sess, config=config)

        self.x_ent_pred = None
        self.entity_emb_ph = None

    def _build_coref_head(self):
        x_ent_train, self.num_entities = self._get_entities_representation(bert_out=self.bert_out_train)
        self.logits_train = self._get_entity_pairs_logits(x_ent_train)
        self.x_ent_pred, _ = self._get_entities_representation(bert_out=self.bert_out_pred)
        self.logits_inference = self._get_entity_pairs_logits(self.x_ent_pred)
        self.labels_pred = tf.argmax(self.logits_inference, axis=-1)  # [batch_size, num_entities]

    def _set_placeholders(self):
        super()._set_placeholders()
        bert_dim = self.config["model"]["bert"]["params"]["hidden_size"]
        multiple = 2 + int(self.config["model"]["coref"]["use_attn"])
        self.entity_emb_ph = tf.placeholder(tf.float32, shape=[None, None, bert_dim * multiple])

    def predict(self, examples: List[Example], flat_chains: bool = True, **kwargs) -> None:
        batch_size = 16
        window = self.config["inference"]["window"]
        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            chunks_batch = []
            id2embeddings = {}
            id_to_num_sentences = {}
            example_ids = []
            for x in examples_batch:
                chunks_batch += x.chunks
                id2embeddings[x.id] = {}  # (start, end) -> np.array размерности D
                example_ids.append(x.id)
            gen = batches_gen(examples=chunks_batch, max_tokens_per_batch=10000, pieces_level=True)
            for batch in gen:
                feed_dict = self._get_feed_dict(batch, mode=ModeKeys.TEST)
                x_ent_pred = self.sess.run(self.x_ent_pred, feed_dict=feed_dict)  # [num_chunks, E, D]
                for i in range(len(batch)):
                    chunk = batch[i]
                    num_sentences = id_to_num_sentences[chunk.parent]
                    end_rel = chunk.tokens[-1].id_sent - chunk.tokens[0].id_sent
                    assert end_rel < window, f"[{chunk.id}] relative end {end_rel} >= window size {window}"
                    is_first = chunk.tokens[0].id_sent == 0
                    is_last = chunk.tokens[-1].id_sent == num_sentences - 1
                    sent_ids = get_sent_ids_to_predict_for(is_first=is_first, is_last=is_last, window=window)

                    index2entity = {entity.index: entity for entity in chunk.entities}
                    for j in range(len(chunk.entities)):
                        entity = index2entity[j]
                        id_sent_abs = entity.tokens[0].id_sent
                        id_sent_rel = id_sent_abs - chunk.tokens[0].id_sent
                        if id_sent_rel in sent_ids:
                            span = entity.tokens[0].index_abs, entity.tokens[-1].index_abs
                            id2embeddings[chunk.parent][span] = x_ent_pred[i, j, :]

            entities_emb = self._agg_embeddings(id2embeddings, example_ids)
            re_labels_pred, re_logits_pred = self.sess.run(
                [self.labels_pred, self.logits_inference],
                feed_dict={self.entity_emb_ph: entities_emb}
            )

            for i, x in enumerate(examples_batch):
                head2dep = {}
                index2entity = {}
                for j, entity in enumerate(sorted(x.entities, key=lambda e: (e.tokens[0].index_rel, e.tokens[-1].index_rel))):
                    index2entity[j] = entity
                for idx_head in range(len(x.entities)):
                    idx_dep = re_labels_pred[i, idx_head]
                    # нет исходящего ребра или находится дальше по тексту
                    if idx_dep == 0 or idx_dep >= idx_head + 1:
                        continue
                    score = re_logits_pred[i, idx_head, idx_dep]
                    # это условие акутально только тогда,
                    # когда реализована оригинальная логика: s(i, eps) = 0
                    # if score <= 0.0:
                    #     continue
                    head = index2entity[idx_head]
                    dep = index2entity[idx_dep - 1]
                    key_head = x.id, head.id
                    if key_head in head2dep:
                        if head2dep[key_head]["score"] < score:
                            head2dep[key_head] = {"dep": dep.id, "score": score}
                        else:
                            pass
                    else:
                        head2dep[key_head] = {"dep": dep.id, "score": score}

                # присовение id_chain
                id2entity = {}
                g = {}
                for entity in x.entities:
                    g[entity.id] = set()
                    id2entity[entity.id] = entity
                for entity in x.entities:
                    key = x.id, entity.id
                    if key in head2dep:
                        id_dep = head2dep[key]["dep"]
                        g[entity.id].add(id_dep)
                        if not flat_chains:
                            id_arc = "R" + str(len(x.arcs))
                            arc = Arc(id=id_arc, head=entity.id, dep=id_dep, rel=self.coref_rel)
                            x.arcs.append(arc)

                components = get_connected_components(g)

                for id_chain, comp in enumerate(components):
                    entities_comp = []
                    for id_entity in comp:
                        entity = id2entity[id_entity]
                        entity.id_chain = id_chain
                        entities_comp.append(entity)
                    if flat_chains and len(comp) > 1:
                        entities_comp_sorted = sorted(entities_comp,
                                                      key=lambda e: (e.tokens[0].index_abs, e.tokens[-1].index_abs))
                        for i in range(len(comp) - 1):
                            dep = entities_comp_sorted[i]
                            head = entities_comp_sorted[i + 1]
                            id_arc = "R" + str(len(x.arcs))
                            arc = Arc(id=id_arc, head=head.id, dep=dep.id, rel=self.coref_rel)
                            x.arcs.append(arc)

    def _agg_embeddings(self, id2embeddings, example_ids):
        num_entities = {}
        for i in example_ids:
            num_entities[i] = len(id2embeddings[i])
        m = max(num_entities.values())
        bert_dim = self.config["model"]["bert"]["params"]["hidden_size"]
        multiple = 2 + int(self.config["model"]["coref"]["use_attn"])
        d = bert_dim * multiple
        zeros = np.zeros(d, dtype=np.float32)
        res = []
        for i in example_ids:
            res_i = []
            for _, v in sorted(id2embeddings[i].items(), key=lambda x: x[0]):
                res_i.append(v[None])
            for _ in range(m - num_entities[i]):
                res_i.append(zeros[None])
            res_i = np.concatenate(res_i, axis=0)  # [num_entities_max, d]
            res.append(res_i[None, :, :])
        res = np.concatenate(res, axis=0)  # [batch_size, num_entities_max, d]
        return res
