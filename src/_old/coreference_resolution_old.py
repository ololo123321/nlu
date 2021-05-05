import random
import os
import json
import shutil
from typing import Dict, List, Callable, Tuple, Iterable
from itertools import chain, groupby, combinations
from abc import ABC, abstractmethod
from collections import defaultdict

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer

from src.model.utils import (
    get_batched_coords_from_labels,
    get_labels_mask,
    get_dense_labels_from_indices,
    get_entity_embeddings,
    get_padded_coords_3d,
    upper_triangular,
    get_entity_embeddings_concat_half,
    get_entity_pairs_mask,
    get_sent_pairs_to_predict_for,
    get_span_indices,
    get_additive_mask
)
from src.model.layers import GraphEncoder, GraphEncoderInputs, StackedBiRNN, MLP
from src.data.base import Example, Arc, Entity
from src.data.postprocessing import get_valid_spans
from src.data.io import to_conll
from src.metrics import (
    classification_report,
    classification_report_ner,
    f1_precision_recall_support,
    get_coreferense_resolution_metrics
)
from src.utils import get_entity_spans, get_connected_components, parse_conll_metrics, train_test_split


class BertForCoreferenceResolution(BertJointModelWithNestedNer):
    """
    * между узлами может быть ровно один тип связи (кореференция)
    * у одного head может быть ровно один dep
    """
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.ner_logits_inference = None
        self.tokens_pair_enc = None

    def _build_graph(self):
        """
        добавлена только эта строчка:
        re_logits_true_entities = tf.squeeze(re_logits_true_entities, axis=[-1])
        :return:
        """
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

                self.ner_logits_train = self._build_ner_head(bert_out=bert_out_train)
                self.ner_logits_inference = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    num_entities = self.config["model"]["ner"]["biaffine"]["num_labels"]
                    if self.config["model"]["re"]["use_birnn"]:
                        emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                    else:
                        emb_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_entities, emb_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                shape = tf.shape(self.ner_logits_train)[:-1]
                no_entity_id = self.config["model"]["ner"]["no_entity_id"]
                ner_labels_dense = get_dense_labels_from_indices(
                    indices=self.ner_labels_ph, shape=shape, no_label_id=no_entity_id
                )

                ner_preds_inference = tf.argmax(self.ner_logits_inference, axis=-1, output_type=tf.int32)

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=ner_labels_dense
                )

                re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_labels_dense
                )
                re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_preds_inference
                )

                # squeeze
                self.re_logits_train = tf.squeeze(self.re_logits_train, axis=[-1])  # [batch_size, num_entities, num_entities]
                re_logits_true_entities = tf.squeeze(re_logits_true_entities, axis=[-1])  # [batch_size, num_entities, num_entities]

                # mask
                self.re_logits_train = self._mask_logits(self.re_logits_train, self.num_entities)
                re_logits_true_entities = self._mask_logits(re_logits_true_entities, self.num_entities)
                re_logits_pred_entities = self._mask_logits(re_logits_pred_entities, self.num_entities_pred)

                # argmax
                self.re_labels_true_entities = tf.argmax(re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(re_logits_pred_entities, axis=-1)

                self.re_logits_true_entities = re_logits_true_entities  # debug TODO: удалить

            self._set_loss()
            self._set_train_op()

    # TODO: вынести в utils
    @staticmethod
    def _mask_logits(logits, num_entities):
        mask = tf.sequence_mask(num_entities, maxlen=tf.shape(logits)[1], dtype=tf.float32)
        # TODO: оставить только одну из двух масок (вроде, вторую)
        logits -= (1.0 - mask[:, :, None]) * 1e9
        logits -= (1.0 - mask[:, None, :]) * 1e9
        return logits

    # TODO: копипаста
    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        """
        адаптирована только часть с arcs_pred
        """
        y_true_ner = []
        y_pred_ner = []

        y_true_re = []
        y_pred_re = []

        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        loss = 0.0
        loss_ner = 0.0
        loss_re = 0.0
        num_batches = 0

        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict = self._get_feed_dict(examples_batch, training=False)
            loss_i, loss_ner_i, loss_re_i, ner_logits, num_entities, re_labels_pred, re_logits = self.sess.run([
                self.loss,
                self.loss_ner,
                self.loss_re,
                self.ner_logits_inference,
                self.num_entities,
                self.re_labels_true_entities,
                self.re_logits_true_entities
            ], feed_dict=feed_dict)
            loss += loss_i
            loss_ner += loss_ner_i
            loss_re += loss_re_i

            # re_labels_pred: [batch_size, num_entities]

            for i, x in enumerate(examples_batch):
                # ner
                num_tokens = len(x.tokens)
                spans_true = np.full((num_tokens, num_tokens), no_entity_id, dtype=np.int32)

                for entity in x.entities:
                    start = entity.tokens[0].index_rel
                    end = entity.tokens[-1].index_rel
                    spans_true[start, end] = entity.label_id

                spans_pred = np.full((num_tokens, num_tokens), no_entity_id, dtype=np.int32)
                ner_logits_i = ner_logits[i, :num_tokens, :num_tokens, :]
                spans_filtered = get_valid_spans(logits=ner_logits_i,  is_flat_ner=False)
                for span in spans_filtered:
                    spans_pred[span.start, span.end] = span.label

                y_true_ner += [self.inv_ner_enc[j] for j in spans_true.flatten()]
                y_pred_ner += [self.inv_ner_enc[j] for j in spans_pred.flatten()]

                # re
                num_entities_i = num_entities[i]
                assert num_entities_i == len(x.entities)
                arcs_true = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    arcs_true[arc.head_index, arc.dep_index] = arc.rel_id

                arcs_pred = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)
                for id_head, id_dep in enumerate(re_labels_pred[i, :num_entities_i]):
                    if id_head != id_dep:
                        try:
                            arcs_pred[id_head, id_dep] = 1
                        except IndexError as e:
                            print("id head:", i)
                            print("id dep:", id_dep)
                            print("num entities i:", num_entities_i)
                            print("re labels:")
                            print(re_labels_pred)
                            print("re labels i:")
                            print(re_labels_pred[i])
                            print("re logits:")
                            print(re_logits)
                            print("re logits i:")
                            print(re_logits[i])
                            print("num entities batch:")
                            print(num_entities)
                            print([len(x.entities) for x in examples_batch])
                            print("examples batch:")
                            print([x.id for x in examples_batch])
                            raise e

                y_true_re += [self.inv_re_enc[j] for j in arcs_true.flatten()]
                y_pred_re += [self.inv_re_enc[j] for j in arcs_pred.flatten()]

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches
        loss_ner /= num_batches
        loss_re /= num_batches

        # ner
        trivial_label = self.inv_ner_enc[no_entity_id]
        ner_metrics = classification_report(y_true=y_true_ner, y_pred=y_pred_ner, trivial_label=trivial_label)

        # re
        re_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, trivial_label="O")

        # total TODO: копипаста из родительского класса
        # сделано так, чтобы случайный скор на таске с нулевым loss_coef не вносил подгрешность в score.
        # невозможность равенства нулю коэффициентов при лоссах на обоих тасках рассмотрена в BaseModel.__init__
        if self.config["model"]["ner"]["loss_coef"] == 0.0:
            score = re_metrics["micro"]["f1"]
        elif self.config["model"]["re"]["loss_coef"] == 0.0:
            score = ner_metrics["micro"]["f1"]
        else:
            score = ner_metrics["micro"]["f1"] * 0.5 + re_metrics["micro"]["f1"] * 0.5

        performance_info = {
            "ner": {
                "loss": loss_ner,
                "metrics": ner_metrics
            },
            "re": {
                "loss": loss_re,
                "metrics": re_metrics,
            },
            "loss": loss,
            "score": score
        }

        return performance_info

    def _get_re_loss(self):
        no_rel_id = self.config["model"]["re"]["no_relation_id"]
        logits_shape = tf.shape(self.re_logits_train)
        labels_shape = logits_shape[:2]
        labels = get_dense_labels_from_indices(
            indices=self.re_labels_ph, shape=labels_shape, no_label_id=no_rel_id
        )  # [batch_size, num_entities]
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.re_logits_train
        )  # [batch_size, num_entities]

        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)

        masked_per_example_loss = per_example_loss * sequence_mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(sequence_mask), tf.float32)
        num_pairs = tf.maximum(num_pairs, 1.0)
        loss = total_loss / num_pairs
        loss *= self.config["model"]["re"]["loss_coef"]
        return loss

    def _set_placeholders(self):
        # bert inputs
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        # ner inputs
        self.first_pieces_coords_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, None, 2], name="first_pieces_coords"
        )  # [id_example, id_piece]
        self.num_pieces_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_pieces")
        self.num_tokens_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_tokens")
        self.ner_labels_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, 4], name="ner_labels"
        )  # [id_example, start, end, label]

        # re
        self.re_labels_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, 3], name="re_labels"
        )  # [id_example, id_head, id_dep]

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    # TODO: много копипасты!
    def _get_feed_dict(self, examples: List[Example], training: bool):
        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

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
            id2entity = {entity.id: entity for entity in x.entities}
            head2dep = {arc.head: id2entity[arc.dep] for arc in x.arcs}

            for entity in x.entities:
                start = entity.tokens[0].index_rel
                end = entity.tokens[-1].index_rel
                label = entity.label_id
                assert isinstance(label, int)
                ner_labels.append((i, start, end, label))

                if entity.id in head2dep.keys():
                    dep_index = head2dep[entity.id].index
                else:
                    dep_index = entity.index
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

        if len(ner_labels) == 0:
            ner_labels.append((0, 0, 0, 0))

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0))

        d = {
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,
            self.first_pieces_coords_ph: first_pieces_coords,
            self.num_pieces_ph: num_pieces,
            self.num_tokens_ph: num_tokens,
            self.ner_labels_ph: ner_labels,
            self.re_labels_ph: re_labels,
            self.training_ph: training
        }
        return d


# TODO: переименовать в BertForCoreferenceResolutionMentionPair
class BertForCoreferenceResolutionV2(BertForCoreferenceResolution):
    """
    учится предсказывать родителя для каждого узла
    """
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.root_emb = None
        self.re_logits_true_entities = None
        self.re_logits_pred_entities = None

    def _build_graph(self):
        """
        добавлен self.root_emb
        """
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

                self.ner_logits_train = self._build_ner_head(bert_out=bert_out_train)
                self.ner_logits_inference = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    num_entities = self.config["model"]["ner"]["biaffine"]["num_labels"]
                    if self.config["model"]["re"]["use_birnn"]:
                        emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                    else:
                        emb_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_entities, emb_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                if self.config["model"]["re"]["use_birnn"]:
                    emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                else:
                    emb_dim = self.config["model"]["bert"]["dim"]
                self.root_emb = tf.get_variable("root_emb", shape=[1, emb_dim], dtype=tf.float32)

                shape = tf.shape(self.ner_logits_train)[:-1]
                no_entity_id = self.config["model"]["ner"]["no_entity_id"]
                ner_labels_dense = get_dense_labels_from_indices(
                    indices=self.ner_labels_ph, shape=shape, no_label_id=no_entity_id
                )

                ner_preds_inference = tf.argmax(self.ner_logits_inference, axis=-1, output_type=tf.int32)

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=ner_labels_dense
                )

                self.re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_labels_dense
                )
                self.re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_preds_inference
                )

                # argmax
                self.re_labels_true_entities = tf.argmax(self.re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(self.re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def _build_re_head(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x, num_entities = self._get_entities_representation(bert_out=bert_out, ner_labels=ner_labels)

        # добавление root
        batch_size = tf.shape(x)[0]
        x_root = tf.tile(self.root_emb, [batch_size, 1])
        x_root = x_root[:, None, :]
        x_dep = tf.concat([x_root, x], axis=1)  # [batch_size, num_entities + 1, bert_dim]

        # encoding of pairs
        inputs = GraphEncoderInputs(head=x, dep=x_dep)
        logits = self.entity_pairs_enc(inputs=inputs, training=self.training_ph)  # [N, num_ent, num_ent + 1, 1]

        # squeeze
        logits = tf.squeeze(logits, axis=[-1])  # [batch_size, num_entities, num_entities + 1]

        # mask
        num_entities_inner = num_entities + tf.ones_like(num_entities)
        mask = tf.sequence_mask(num_entities_inner)
        logits += get_additive_mask(mask[:, None, :])

        return logits, num_entities

    # TODO: много копипасты!
    def _get_feed_dict(self, examples: List[Example], mode: str):
        assert self.ner_enc is not None
        assert self.re_enc is not None

        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

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
                id2entity = {entity.id: entity for entity in x.entities}
                head2dep = {arc.head: id2entity[arc.dep] for arc in x.arcs}

                for entity in x.entities:
                    start = entity.tokens[0].index_rel
                    end = entity.tokens[-1].index_rel
                    id_label = self.ner_enc[entity.label]
                    ner_labels.append((i, start, end, id_label))

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

        if len(ner_labels) == 0:
            ner_labels.append((0, 0, 0, 0))

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
            d[self.ner_labels_ph] = ner_labels
            d[self.re_labels_ph] = re_labels

        return d

    # TODO: копипаста
    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        """
        адаптирована только часть с arcs_pred
        """
        y_true_ner = []
        y_pred_ner = []

        y_true_re = []
        y_pred_re = []

        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        loss = 0.0
        loss_ner = 0.0
        loss_re = 0.0
        num_batches = 0

        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict = self._get_feed_dict(examples_batch, mode=ModeKeys.VALID)
            loss_i, loss_ner_i, loss_re_i, ner_logits, num_entities, re_labels_pred = self.sess.run([
                self.loss,
                self.loss_ner,
                self.loss_re,
                self.ner_logits_inference,
                self.num_entities,
                self.re_labels_true_entities,
            ], feed_dict=feed_dict)
            loss += loss_i
            loss_ner += loss_ner_i
            loss_re += loss_re_i

            for i, x in enumerate(examples_batch):
                # ner
                num_tokens = len(x.tokens)
                spans_true = np.full((num_tokens, num_tokens), no_entity_id, dtype=np.int32)

                for entity in x.entities:
                    start = entity.tokens[0].index_rel
                    end = entity.tokens[-1].index_rel
                    spans_true[start, end] = self.ner_enc[entity.label]

                spans_pred = np.full((num_tokens, num_tokens), no_entity_id, dtype=np.int32)
                ner_logits_i = ner_logits[i, :num_tokens, :num_tokens, :]
                spans_filtered = get_valid_spans(logits=ner_logits_i,  is_flat_ner=False)
                for span in spans_filtered:
                    spans_pred[span.start, span.end] = span.label

                y_true_ner += [self.inv_ner_enc[j] for j in spans_true.flatten()]
                y_pred_ner += [self.inv_ner_enc[j] for j in spans_pred.flatten()]

                # re
                num_entities_i = num_entities[i]
                assert num_entities_i == len(x.entities)
                arcs_true = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    arcs_true[arc.head_index, arc.dep_index] = self.re_enc[arc.rel]

                arcs_pred = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)
                for id_head, id_dep in enumerate(re_labels_pred[i, :num_entities_i]):
                    if id_dep != 0:
                        try:
                            arcs_pred[id_head, id_dep - 1] = 1
                        except IndexError as e:
                            print("i:", i)
                            print("id head:", id_head)
                            print("id dep:", id_dep)
                            print("num entities i:", num_entities_i)
                            print("re labels:")
                            print(re_labels_pred)
                            print("re labels i:")
                            print(re_labels_pred[i])
                            # print("re logits:")
                            # print(re_logits)
                            # print("re logits i:")
                            # print(re_logits[i])
                            print("num entities batch:")
                            print(num_entities)
                            print([len(x.entities) for x in examples_batch])
                            print("examples batch:")
                            print([x.id for x in examples_batch])
                            raise e

                y_true_re += [self.inv_re_enc[j] for j in arcs_true.flatten()]
                y_pred_re += [self.inv_re_enc[j] for j in arcs_pred.flatten()]

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches
        loss_ner /= num_batches
        loss_re /= num_batches

        # ner
        trivial_label = self.inv_ner_enc[no_entity_id]
        ner_metrics = classification_report(y_true=y_true_ner, y_pred=y_pred_ner, trivial_label=trivial_label)

        # re
        re_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, trivial_label="O")

        # total TODO: копипаста из родительского класса
        # сделано так, чтобы случайный скор на таске с нулевым loss_coef не вносил подгрешность в score.
        # невозможность равенства нулю коэффициентов при лоссах на обоих тасках рассмотрена в BaseModel.__init__
        if self.config["model"]["ner"]["loss_coef"] == 0.0:
            score = re_metrics["micro"]["f1"]
        elif self.config["model"]["re"]["loss_coef"] == 0.0:
            score = ner_metrics["micro"]["f1"]
        else:
            score = ner_metrics["micro"]["f1"] * 0.5 + re_metrics["micro"]["f1"] * 0.5

        performance_info = {
            "ner": {
                "loss": loss_ner,
                "metrics": ner_metrics
            },
            "re": {
                "loss": loss_re,
                "metrics": re_metrics,
            },
            "loss": loss,
            "score": score
        }

        return performance_info

    # TODO: копипаста
    # TODO: пока в _get_feed_dict суётся ModeKeys.VALID, потому что предполагается,
    #  что сущности уже найдены. избавиться от этого костыля, путём создания отдельного
    #  класса под модель, где нужно искать только кореференции
    def predict(
            self,
            examples: List[Example],
            chunks: List[Example],
            window: int = 1,
            batch_size: int = 16,
            no_or_one_parent_per_node: bool = False,
            **kwargs
    ):
        """
        Оценка качества на уровне документа.
        :param examples: документы
        :param chunks: куски (stride 1). предполагаетя, что для каждого документа из examples должны быть куски в chunks
        :param window: размер кусков (в предложениях)
        :param batch_size:
        :param no_or_one_parent_per_node
        :return:
        """
        # проверка на то, то в примерах нет рёбер
        # TODO: как-то обработать случай отсутствия сущнсоетй
        for x in examples:
            assert len(x.arcs) == 0

        id_to_num_sentences = {x.id: x.tokens[-1].id_sent + 1 for x in examples}

        # y_pred = set()
        head2dep = {}  # (file, head) -> {dep, score}
        dep2head = {}

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunks_batch = chunks[start:end]
            feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.VALID)
            re_labels_pred, re_logits_pred = self.sess.run(
                [self.re_labels_true_entities, self.re_logits_true_entities],
                feed_dict=feed_dict
            )
            # re_labels_pred: np.ndarray, shape [batch_size, num_entities], dtype np.int32
            # values in range [0, num_ent]; 0 means no dep.
            # re_logits_pred: np.ndarray, shape [batch_size, num_entities, num_entities + 1], dtype np.float32

            for i in range(len(chunks_batch)):
                chunk = chunks_batch[i]

                num_entities_chunk = len(chunk.entities)
                # entities_sorted = sorted(chunk.entities, key=lambda e: (e.tokens[0].index_rel, e.tokens[-1].index_rel))
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
                        # нет исходящего ребра
                        if idx_dep == 0:
                            continue
                        # петля
                        if idx_head == idx_dep - 1:
                            continue
                        # head = entities_sorted[idx_head]
                        # dep = entities_sorted[idx_dep - 1]
                        head = index2entity[idx_head]
                        dep = index2entity[idx_dep - 1]
                        id_sent_head = head.tokens[0].id_sent
                        id_sent_dep = dep.tokens[0].id_sent
                        if (id_sent_head == id_sent_abs_a and id_sent_dep == id_sent_abs_b) or \
                                (id_sent_head == id_sent_abs_b and id_sent_dep == id_sent_abs_a):
                            score = re_logits_pred[i, idx_head, idx_dep]
                            key_head = chunk.parent, head.id
                            key_dep = chunk.parent, dep.id
                            if key_head in head2dep:
                                if head2dep[key_head]["score"] < score:
                                    head2dep[key_head] = {"dep": dep.id, "score": score}
                                else:
                                    pass
                            else:
                                if no_or_one_parent_per_node:
                                    if key_dep in dep2head:
                                        if dep2head[key_dep]["score"] < score:
                                            dep2head[key_dep] = {"head": head.id, "score": score}
                                            head2dep.pop(key_head, None)
                                        else:
                                            pass
                                    else:
                                        dep2head[key_dep] = {"head": head.id, "score": score}
                                        head2dep[key_head] = {"dep": dep.id, "score": score}
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
                    dep = head2dep[key]["dep"]
                    g[entity.id].add(dep)
                    id_arc = "R" + str(len(x.arcs))
                    arc = Arc(id=id_arc, head=entity.id, dep=dep, rel=self.inv_re_enc[1])
                    x.arcs.append(arc)

            components = get_connected_components(g)

            for id_chain, comp in enumerate(components):
                for id_entity in comp:
                    id2entity[id_entity].id_chain = id_chain


class BertForCoreferenceResolutionV21(BertForCoreferenceResolutionV2):
    """
    + один таргет: (i, j) -> {1, если сущности i и j относятся к одному кластеру, 0 - иначе}
    процедура инференса не меняется
    """
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.entity_pairs_enc2 = None
        self.re_labels2_ph = None
        self.re_logits2_train = None

    def _build_graph(self):
        """
        добавлен self.root_emb
        """
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

                self.ner_logits_train = self._build_ner_head(bert_out=bert_out_train)
                self.ner_logits_inference = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    num_entities = self.config["model"]["ner"]["biaffine"]["num_labels"]
                    if self.config["model"]["re"]["use_birnn"]:
                        emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                    else:
                        emb_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_entities, emb_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                with tf.variable_scope("re_head_second"):
                    self.entity_pairs_enc2 = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                if self.config["model"]["re"]["use_birnn"]:
                    emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                else:
                    emb_dim = self.config["model"]["bert"]["dim"]
                self.root_emb = tf.get_variable("root_emb", shape=[1, emb_dim], dtype=tf.float32)

                shape = tf.shape(self.ner_logits_train)[:-1]
                no_entity_id = self.config["model"]["ner"]["no_entity_id"]
                ner_labels_dense = get_dense_labels_from_indices(
                    indices=self.ner_labels_ph, shape=shape, no_label_id=no_entity_id
                )

                ner_preds_inference = tf.argmax(self.ner_logits_inference, axis=-1, output_type=tf.int32)

                # first re head
                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=ner_labels_dense
                )

                self.re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_labels_dense
                )
                self.re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_preds_inference
                )

                # seconds re head
                self.re_logits2_train, _ = self._build_re_head2(
                    bert_out=bert_out_train, ner_labels=ner_labels_dense
                )

                # argmax
                self.re_labels_true_entities = tf.argmax(self.re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(self.re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def _build_re_head2(self, bert_out: tf.Tensor, ner_labels: tf.Tensor):
        x, num_entities = self._get_entities_representation(bert_out=bert_out, ner_labels=ner_labels)

        # encoding of pairs
        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.entity_pairs_enc2(inputs=inputs, training=self.training_ph)  # [N, num_ent, num_ent, 1]

        # squeeze
        logits = tf.squeeze(logits, axis=[-1])  # [N, num_ent, num_ent]
        return logits, num_entities

    def _set_placeholders(self):
        super()._set_placeholders()
        self.re_labels2_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, 4], name="re_labels2"
        )  # [id_example, id_head, id_dep, id_rel]

    def _get_re_loss(self):
        loss1 = super()._get_re_loss()

        no_rel_id = self.config["model"]["re"]["no_relation_id"]
        logits_shape = tf.shape(self.re_logits2_train)  # [4]
        labels_shape = logits_shape[:3]  # [3]
        labels = get_dense_labels_from_indices(
            indices=self.re_labels2_ph,
            shape=labels_shape,
            no_label_id=no_rel_id
        )  # [batch_size, num_entities, num_entities]
        labels = tf.cast(labels, tf.float32)
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=self.re_logits2_train
        )  # [batch_size, num_entities, num_entities]

        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)
        mask = sequence_mask[:, None, :] * sequence_mask[:, :, None]

        masked_per_example_loss = per_example_loss * mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(mask), tf.float32)
        num_pairs = tf.maximum(num_pairs, 1.0)
        loss2 = total_loss / num_pairs
        loss2 *= self.config["model"]["re"]["loss_coef"]

        loss = loss1 + loss2
        return loss

    # TODO: много копипасты!
    def _get_feed_dict(self, examples: List[Example], mode: str):
        assert self.ner_enc is not None
        assert self.re_enc is not None

        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

        # re
        re_labels = []
        re_labels2 = []

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
                id2entity = {}
                for entity in x.entities:
                    assert entity.id_chain is not None
                    id2entity[entity.id] = entity

                id_head_to_dep_index = {}
                for arc in x.arcs:
                    id_head_to_dep_index[arc.head] = id2entity[arc.dep].index

                for entity in x.entities:
                    start = entity.tokens[0].index_rel
                    end = entity.tokens[-1].index_rel
                    id_label = self.ner_enc[entity.label]
                    ner_labels.append((i, start, end, id_label))

                    if entity.id in id_head_to_dep_index.keys():
                        dep_index = id_head_to_dep_index[entity.id] + 1
                    else:
                        dep_index = 0
                    re_labels.append((i, entity.index, dep_index))
                    re_labels2.append((i, entity.index, entity.index, 1))  # сущность сама с собой всегда в одном кластере

                for _, group in groupby(sorted(x.entities, key=lambda e: e.id_chain), key=lambda e: e.id_chain):
                    for head, dep in combinations(group, 2):
                        re_labels2.append((i, head.index, dep.index, 1))
                        re_labels2.append((i, dep.index, head.index, 1))

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

        if len(ner_labels) == 0:
            ner_labels.append((0, 0, 0, 0))

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0))

        if len(re_labels2) == 0:
            re_labels2.append((0, 0, 0, 0))

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
            d[self.ner_labels_ph] = ner_labels
            d[self.re_labels_ph] = re_labels
            d[self.re_labels2_ph] = re_labels2

        return d


# TODO: переименовать в BertForCoreferenceResolutionHigherOrder
class BertForCoreferenceResolutionV3(BertForCoreferenceResolutionV2):
    """
    https://arxiv.org/abs/1804.05392

    + config["model"]["re"]["order"]:
        1 - as in V2
        2 - as in paper
    + config["model"]["re"]["w_dropout"]: dropout for self.w
    + config["model"]["re"]["w_dropout_policy"]:
        0 - same mask on each iteration
        1 - different mask on each iteration
    """
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.w = None
        self.w_dropout = None

    # TODO: копипаста из родительского класса только из-за инициализации self.w и self.w_dropout
    def _build_graph(self):
        """
        добавлен self.root_emb
        """
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

                self.ner_logits_train = self._build_ner_head(bert_out=bert_out_train)
                self.ner_logits_inference = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    num_entities = self.config["model"]["ner"]["biaffine"]["num_labels"]
                    if self.config["model"]["re"]["use_birnn"]:
                        emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                    else:
                        emb_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_entities, emb_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                if self.config["model"]["re"]["use_birnn"]:
                    emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                else:
                    emb_dim = self.config["model"]["bert"]["dim"]
                self.root_emb = tf.get_variable("root_emb", shape=[1, emb_dim], dtype=tf.float32)

                self.w = tf.get_variable("w_update", shape=[emb_dim * 2, emb_dim], dtype=tf.float32)
                self.w_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["w_dropout"])

                shape = tf.shape(self.ner_logits_train)[:-1]
                no_entity_id = self.config["model"]["ner"]["no_entity_id"]
                ner_labels_dense = get_dense_labels_from_indices(
                    indices=self.ner_labels_ph, shape=shape, no_label_id=no_entity_id
                )

                ner_preds_inference = tf.argmax(self.ner_logits_inference, axis=-1, output_type=tf.int32)

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=ner_labels_dense
                )

                self.re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_labels_dense
                )
                self.re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_preds_inference
                )

                # argmax
                self.re_labels_true_entities = tf.argmax(self.re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(self.re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def _build_re_head(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # x - [batch_size, num_entities, bert_dim]
        # num_entities - [batch_size]
        x, num_entities = self._get_entities_representation(bert_out=bert_out, ner_labels=ner_labels)

        batch_size = tf.shape(x)[0]
        x_root = tf.tile(self.root_emb, [batch_size, 1])
        x_root = x_root[:, None, :]

        num_entities_inner = num_entities + tf.ones_like(num_entities)

        def get_logits(enc, g):
            g_dep = tf.concat([x_root, g], axis=1)  # [batch_size, num_entities + 1, bert_dim]

            # encoding of pairs
            inputs = GraphEncoderInputs(head=g, dep=g_dep)
            logits = enc(inputs=inputs, training=self.training_ph)  # [N, num_ent, num_ent + 1, 1]

            # squeeze
            logits = tf.squeeze(logits, axis=[-1])  # [batch_size, num_entities, num_entities + 1]

            # mask
            mask = tf.sequence_mask(num_entities_inner)
            logits += get_additive_mask(mask[:, None, :])  # [batch_size, num_entities, num_entities + 1]

            return g_dep, logits

        # n = 1 - baseline
        # n = 2 - like in paper
        n = self.config["model"]["re"]["order"]

        # 0 - one mask for each iteration
        # 1 - different mask on each iteration
        dropout_policy = self.config["model"]["re"]["w_dropout_policy"]

        if dropout_policy == 0:
            w = self.w_dropout(self.w, training=self.training_ph)
        elif dropout_policy == 1:
            w = self.w
        else:
            raise NotImplementedError

        for i in range(n - 1):
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


class BertForCoreferenceResolutionV4(BertForCoreferenceResolutionV2):
    """
    task:
    для каждого mention предсказывать подмножество antecedents
    из множества ранее упомянутых mentions.

    entity representation:
    g = [x_start, x_end, x_attn]

    coref score:
    biaffine without one component

    loss:
    softmax_loss

    inference:
    s(i, j) = 0, if j >=i and j = 0 (no coref)
    During inference, the model only creates a link if the highest antecedent score is positive.
    """
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.dense_attn_1 = None
        self.dense_attn_2 = None

    # TODO: копипаста из родительского класса из-за 1) инициализации ffnn_attn, 2) root_emb -> bert_dim * 3
    def _build_graph(self):
        """
        добавлен self.root_emb
        """
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

                self.ner_logits_train = self._build_ner_head(bert_out=bert_out_train)
                self.ner_logits_inference = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    num_entities = self.config["model"]["ner"]["biaffine"]["num_labels"]
                    if self.config["model"]["re"]["use_birnn"]:
                        emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                    else:
                        emb_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_entities, emb_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                if self.config["model"]["re"]["use_birnn"]:
                    emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                else:
                    emb_dim = self.config["model"]["bert"]["dim"]
                self.root_emb = tf.get_variable("root_emb", shape=[1, emb_dim * 3], dtype=tf.float32)

                # self.w = tf.get_variable("w_update", shape=[emb_dim * 6, emb_dim * 3], dtype=tf.float32)
                # self.w_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["w_dropout"])

                # TODO: вынести гиперпараметры в конфиг!!1!
                self.dense_attn_1 = MLP(num_layers=1, hidden_dim=128, activation=tf.nn.relu, dropout=0.33)
                self.dense_attn_2 = MLP(num_layers=1, hidden_dim=1, activation=None, dropout=None)

                shape = tf.shape(self.ner_logits_train)[:-1]
                no_entity_id = self.config["model"]["ner"]["no_entity_id"]
                ner_labels_dense = get_dense_labels_from_indices(
                    indices=self.ner_labels_ph, shape=shape, no_label_id=no_entity_id
                )

                ner_preds_inference = tf.argmax(self.ner_logits_inference, axis=-1, output_type=tf.int32)

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=ner_labels_dense
                )

                self.re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_labels_dense
                )
                self.re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_preds_inference
                )

                # argmax
                self.re_labels_true_entities = tf.argmax(self.re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(self.re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def _get_entities_representation(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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

        # маскирование
        num_tokens = tf.shape(ner_labels)[1]
        mask = upper_triangular(num_tokens, dtype=tf.int32)
        ner_labels *= mask[None, :, :]

        # векторизация сущностей
        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        span_mask = tf.not_equal(ner_labels, no_entity_id)  # [batch_size, num_tokens, num_tokens]
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


# current best model
class BertForCoreferenceResolutionV5(BertForCoreferenceResolutionV2):
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

    def _build_re_head(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x, num_entities = self._get_entities_representation(bert_out=bert_out, ner_labels=ner_labels)

        # добавление root
        batch_size = tf.shape(x)[0]
        x_root = tf.tile(self.root_emb, [batch_size, 1])
        x_root = x_root[:, None, :]
        x_dep = tf.concat([x_root, x], axis=1)  # [batch_size, num_entities + 1, bert_dim]

        # encoding of pairs
        inputs = GraphEncoderInputs(head=x, dep=x_dep)
        logits = self.entity_pairs_enc(inputs=inputs, training=self.training_ph)  # [N, num_ent, num_ent + 1, 1]

        # squeeze
        logits = tf.squeeze(logits, axis=[-1])  # [batch_size, num_entities, num_entities + 1]

        # mask padding
        num_entities_inner = num_entities + tf.ones_like(num_entities)
        mask_pad = tf.sequence_mask(num_entities_inner)  # [batch_size, num_entities + 1]

        # mask antecedent
        n = tf.shape(logits)[1]
        mask_ant = tf.linalg.band_part(tf.ones((n, n + 1), dtype=tf.bool), -1, 0)  # lower-triangular

        mask = tf.logical_and(mask_pad[:, None, :], mask_ant[None, :, :])
        logits += get_additive_mask(mask)
        return logits, num_entities

    # TODO: копипаста
    # TODO: пока в _get_feed_dict суётся ModeKeys.VALID, потому что предполагается,
    #  что сущности уже найдены. избавиться от этого костыля, путём создания отдельного
    #  класса под модель, где нужно искать только кореференции
    def predict(
            self,
            examples: List[Example],
            chunks: List[Example],
            window: int = 1,
            batch_size: int = 16,
            no_or_one_parent_per_node: bool = False,
            **kwargs
    ):
        """
        Оценка качества на уровне документа.
        :param examples: документы
        :param chunks: куски (stride 1). предполагаетя, что для каждого документа из examples должны быть куски в chunks
        :param window: размер кусков (в предложениях)
        :param batch_size:
        :param no_or_one_parent_per_node
        :return:
        """
        # проверка на то, то в примерах нет рёбер
        # TODO: как-то обработать случай отсутствия сущнсоетй
        for x in examples:
            assert len(x.arcs) == 0

        id_to_num_sentences = {x.id: x.tokens[-1].id_sent + 1 for x in examples}

        # y_pred = set()
        head2dep = {}  # (file, head) -> {dep, score}
        dep2head = {}

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunks_batch = chunks[start:end]
            feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.VALID)
            re_labels_pred, re_logits_pred = self.sess.run(
                [self.re_labels_true_entities, self.re_logits_true_entities],
                feed_dict=feed_dict
            )
            # re_labels_pred: np.ndarray, shape [batch_size, num_entities], dtype np.int32
            # values in range [0, num_ent]; 0 means no dep.
            # re_logits_pred: np.ndarray, shape [batch_size, num_entities, num_entities + 1], dtype np.float32

            for i in range(len(chunks_batch)):
                chunk = chunks_batch[i]

                num_entities_chunk = len(chunk.entities)
                # entities_sorted = sorted(chunk.entities, key=lambda e: (e.tokens[0].index_rel, e.tokens[-1].index_rel))
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
                        # head = entities_sorted[idx_head]
                        # dep = entities_sorted[idx_dep - 1]
                        head = index2entity[idx_head]
                        dep = index2entity[idx_dep - 1]
                        id_sent_head = head.tokens[0].id_sent
                        id_sent_dep = dep.tokens[0].id_sent
                        if (id_sent_head == id_sent_abs_a and id_sent_dep == id_sent_abs_b) or \
                                (id_sent_head == id_sent_abs_b and id_sent_dep == id_sent_abs_a):
                            score = re_logits_pred[i, idx_head, idx_dep]
                            key_head = chunk.parent, head.id
                            key_dep = chunk.parent, dep.id
                            if key_head in head2dep:
                                if head2dep[key_head]["score"] < score:
                                    head2dep[key_head] = {"dep": dep.id, "score": score}
                                else:
                                    pass
                            else:
                                if no_or_one_parent_per_node:
                                    if key_dep in dep2head:
                                        if dep2head[key_dep]["score"] < score:
                                            dep2head[key_dep] = {"head": head.id, "score": score}
                                            head2dep.pop(key_head, None)
                                        else:
                                            pass
                                    else:
                                        dep2head[key_dep] = {"head": head.id, "score": score}
                                        head2dep[key_head] = {"dep": dep.id, "score": score}
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
                    dep = head2dep[key]["dep"]
                    g[entity.id].add(dep)
                    id_arc = "R" + str(len(x.arcs))
                    arc = Arc(id=id_arc, head=entity.id, dep=dep, rel=self.inv_re_enc[1])
                    x.arcs.append(arc)

            components = get_connected_components(g)

            for id_chain, comp in enumerate(components):
                for id_entity in comp:
                    id2entity[id_entity].id_chain = id_chain


class BertForCoreferenceResolutionV51(BertForCoreferenceResolutionV5):
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.examples_valid = None
        self.examples_valid_copy = None

    # TODO: копипаста из V6
    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        assert self.examples_valid is not None
        assert self.examples_valid_copy is not None

        for x in self.examples_valid_copy:
            x.arcs = []
            for entity in x.entities:
                entity.id_chain = None

        self.predict(
            examples=self.examples_valid_copy,
            chunks=examples,
            batch_size=batch_size,
            window=self.config["valid"]["window"],
        )

        to_conll(
            examples=self.examples_valid,
            path=self.config["valid"]["path_true"]
        )

        to_conll(
            examples=self.examples_valid_copy,
            path=self.config["valid"]["path_pred"]
        )

        metrics = {}
        for metric in ["muc", "bcub", "ceafm", "ceafe", "blanc"]:
            stdout = get_coreferense_resolution_metrics(
                path_true=self.config["valid"]["path_true"],
                path_pred=self.config["valid"]["path_pred"],
                scorer_path=self.config["valid"]["scorer_path"],
                metric=metric
            )
            # print(metric)
            # print(stdout)
            is_blanc = metric == "blanc"
            metrics[metric] = parse_conll_metrics(stdout=stdout, is_blanc=is_blanc)

        score = (metrics["muc"]["f1"] + metrics["bcub"]["f1"] + metrics["ceafm"]["f1"] + metrics["ceafe"]["f1"]) / 4.0
        d = {
            "loss": 0.0,
            "score": score,
            "metrics": metrics
        }
        return d


# TODO: s(i, eps) = 0
class BertForCoreferenceResolutionV6(BertForCoreferenceResolutionV5):
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.examples_valid = None
        self.examples_valid_copy = None

        # для отладки переобучения
        self.examples_test = None
        self.examples_test_copy = None
        self.chunks_test = None

    def predict(
            self,
            examples: List[Example],
            chunks: List[Example],
            window: int = 1,
            batch_size: int = 16,
            no_or_one_parent_per_node: bool = False,
            **kwargs
    ):
        """
        Оценка качества на уровне документа.
        :param examples: документы
        :param chunks: куски (stride 1). предполагаетя, что для каждого документа из examples должны быть куски в chunks
        :param window: размер кусков (в предложениях)
        :param batch_size:
        :param no_or_one_parent_per_node
        :return:
        """
        # проверка на то, то в примерах нет рёбер
        # TODO: как-то обработать случай отсутствия сущнсоетй
        for x in examples:
            assert len(x.arcs) == 0

        id_to_num_sentences = {x.id: x.tokens[-1].id_sent + 1 for x in examples}

        # y_pred = set()
        head2dep = {}  # (file, head) -> {dep, score}

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunks_batch = chunks[start:end]
            feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.VALID)
            re_labels_pred, re_logits_pred = self.sess.run(
                [self.re_labels_true_entities, self.re_logits_true_entities],
                feed_dict=feed_dict
            )
            # re_labels_pred: np.ndarray, shape [batch_size, num_entities], dtype np.int32
            # values in range [0, num_ent]; 0 means no dep.
            # re_logits_pred: np.ndarray, shape [batch_size, num_entities, num_entities + 1], dtype np.float32

            for i in range(len(chunks_batch)):
                chunk = chunks_batch[i]

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
                    dep = head2dep[key]["dep"]
                    g[entity.id].add(dep)
                    id_arc = "R" + str(len(x.arcs))
                    arc = Arc(id=id_arc, head=entity.id, dep=dep, rel=self.inv_re_enc[1])
                    x.arcs.append(arc)

            # print(g)
            components = get_connected_components(g)

            for id_chain, comp in enumerate(components):
                for id_entity in comp:
                    id2entity[id_entity].id_chain = id_chain

    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        assert self.examples_valid is not None
        assert self.examples_valid_copy is not None

        for x in self.examples_valid_copy:
            x.arcs = []
            for entity in x.entities:
                entity.id_chain = None

        self.predict(
            examples=self.examples_valid_copy,
            chunks=examples,
            batch_size=batch_size,
            window=self.config["valid"]["window"],
        )

        to_conll(
            examples=self.examples_valid,
            path=self.config["valid"]["path_true"]
        )

        to_conll(
            examples=self.examples_valid_copy,
            path=self.config["valid"]["path_pred"]
        )

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
            "loss": 0.0,
            "score": score,
            "metrics": metrics
        }

        if self.chunks_test is not None and self.examples_test is not None and self.examples_test_copy is not None:
            for x in self.examples_test_copy:
                x.arcs = []
                for entity in x.entities:
                    entity.id_chain = None

            self.predict(
                examples=self.examples_test_copy,
                chunks=self.chunks_test,
                batch_size=batch_size,
                window=self.config["valid"]["window"],
            )

            to_conll(
                examples=self.examples_test,
                path=self.config["valid"]["path_true"]
            )

            to_conll(
                examples=self.examples_test_copy,
                path=self.config["valid"]["path_pred"]
            )

            d["test_score"] = 0.0
            for metric in ["muc", "bcub", "ceafm", "ceafe", "blanc"]:
                stdout = get_coreferense_resolution_metrics(
                    path_true=self.config["valid"]["path_true"],
                    path_pred=self.config["valid"]["path_pred"],
                    scorer_path=self.config["valid"]["scorer_path"],
                    metric=metric
                )
                is_blanc = metric == "blanc"
                d["metrics"][metric + "_test"] = parse_conll_metrics(stdout=stdout, is_blanc=is_blanc)
                if metric in {"muc", "bcub", "ceafm", "ceafe"}:
                    d["test_score"] += d["metrics"][metric + "_test"]["f1"] * 0.25

        return d

    def _get_re_loss(self):
        no_rel_id = 0  # должен быть ноль обязательно
        logits_shape = tf.shape(self.re_logits_train)
        ones = tf.ones_like(self.re_labels_ph[:, :1])
        indices = tf.concat([self.re_labels_ph, ones], axis=1)
        labels = get_dense_labels_from_indices(
            indices=indices, shape=logits_shape, no_label_id=no_rel_id
        )  # [batch_size, num_entities, num_entities + 1], {0, 1}

        # предполагается, что логиты уже маскированы по последнему измерению (pad, look-ahead)
        scores_model = tf.reduce_logsumexp(self.re_logits_train, axis=-1)  # [batch_size, num_entities]
        logits_gold = self.re_logits_train + get_additive_mask(labels)  # [batch_size, num_entities, num_entities + 1]
        scores_gold = tf.reduce_logsumexp(logits_gold, axis=-1)  # [batch_size, num_entities]
        per_example_loss = scores_model - scores_gold  # [batch_size, num_entities]

        # mask
        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)
        masked_per_example_loss = per_example_loss * sequence_mask

        # aggregate
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_entities_total = tf.cast(tf.reduce_sum(self.num_entities), tf.float32)
        num_entities_total = tf.maximum(num_entities_total, 1.0)
        loss = total_loss / num_entities_total

        # weight
        loss *= self.config["model"]["re"]["loss_coef"]
        return loss

    # TODO: много копипасты!
    def _get_feed_dict(self, examples: List[Example], mode: str):
        assert self.ner_enc is not None
        assert self.re_enc is not None

        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

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
                    # ner
                    start = entity.tokens[0].index_rel
                    end = entity.tokens[-1].index_rel
                    id_label = self.ner_enc[entity.label]
                    ner_labels.append((i, start, end, id_label))

                    # re
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

        if len(ner_labels) == 0:
            ner_labels.append((0, 0, 0, 0))

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
            d[self.ner_labels_ph] = ner_labels
            d[self.re_labels_ph] = re_labels

        return d
