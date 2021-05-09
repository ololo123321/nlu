import os
import json
from typing import Dict, List
from abc import abstractmethod
from itertools import chain

import tensorflow as tf
import numpy as np

from src.data.base import Example, Entity
from src.data.postprocessing import get_valid_spans
from src.model.base import BaseModelRelationExtraction, BaseModelBert, ModeKeys
from src.model.layers import StackedBiRNN, GraphEncoder, GraphEncoderInputs
from src.model.utils import upper_triangular
from src.metrics import classification_report, classification_report_ner
from src.utils import get_entity_spans, batches_gen, get_filtered_by_length_chunks


class BertForRelationExtraction(BaseModelRelationExtraction, BaseModelBert):
    """
    1. Поиск сущностей и триггеров событий (flat ner)
    2. Поиск отношений между сущностями и аргументов событий

    https://arxiv.org/abs/1812.11275
    """

    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        """
        config = {
            "model": {
                "bert": {
                    "dir": "~/bert",
                    "dim": 768,
                    "attention_probs_dropout_prob": 0.5,  # default 0.1
                    "hidden_dropout_prob": 0.1,
                    "dropout": 0.1,
                    "scope": "bert",
                    "pad_token_id": 0,
                    "cls_token_id": 1,
                    "sep_token_id": 2
                },
                "ner": {
                    "use_crf": True,
                    "num_labels": 7,
                    "no_entity_id": 0,
                    "start_ids": [1, 2, 3],  # id лейблов первых токенов сущностей. нужно для векторизации сущностей
                    "prefix_joiner": "-",
                    "loss_coef": 1.0,
                    "use_birnn": True,
                    "rnn": {
                        "num_layers": 1,
                        "cell_dim": 128,
                        "dropout": 0.5,
                        "recurrent_dropout": 0.0
                    }
                },
                "re": {
                    "no_relation_id": 0,
                    "loss_coef": 10.0,
                    "use_birnn": True,
                    "use_entity_emb": True,
                    "use_entity_emb_layer_norm": True,
                    "entity_emb_dropout": 0.2,
                    "rnn": {
                        "num_layers": 1,
                        "cell_dim": 128,
                        "dropout": 0.5,
                        "recurrent_dropout": 0.0
                    },
                    "biaffine": {
                        "num_mlp_layers": 1,
                        "activation": "relu",
                        "head_dim": 128,
                        "dep_dim": 128,
                        "dropout": 0.3,
                        "num_labels": 7,
                    }
                }
            },
            "training": {
                "num_epochs": 100,
                "batch_size": 16,
                "max_epochs_wo_improvement": 10
            },
            "optimizer": {
                "init_lr": 2e-5,
                "num_train_steps": 100000,
                "num_warmup_steps": 10000
            }
        }
        """
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        # PLACEHOLDERS
        # bert
        self.input_ids_ph = None
        self.input_mask_ph = None
        self.segment_ids_ph = None

        # ner
        self.first_pieces_coords_ph = None
        self.num_pieces_ph = None  # для обучаемых с нуля рекуррентных слоёв
        self.num_tokens_ph = None  # для crf
        self.ner_labels_ph = None

        # re
        self.re_labels_ph = None

        # common
        self.training_ph = None

        # TENSORS
        self.loss_ner = None
        self.loss_re = None
        self.ner_logits_train = None
        self.transition_params = None
        self.ner_preds_inference = None
        self.re_logits_train = None
        self.re_labels_true_entities = None
        self.re_labels_pred_entities = None
        self.num_entities = None
        self.num_entities_pred = None

        # LAYERS
        self.bert_dropout = None
        self.birnn_ner = None
        self.birnn_re = None
        self.dense_ner_labels = None
        self.ner_emb = None
        self.ner_emb_layer_norm = None
        self.ner_emb_dropout = None
        self.entity_pairs_enc = None

        # OPS
        self.train_op_head = None

    def _build_re_head(self):
        # self._set_placeholders()
        #
        # # N - batch size
        # # D - bert dim
        # # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        # with tf.variable_scope(self.model_scope):
        #     bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
        #     bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]
        #
        #     self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])
        #
        #     # ner
        #     with tf.variable_scope(self.ner_scope):
        #         if self.config["model"]["ner"]["use_birnn"]:
        #             self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])
        #
        #         num_labels = self.config["model"]["ner"]["num_labels"]
        #         self.dense_ner_labels = tf.keras.layers.Dense(num_labels)
        #
        #         self.ner_logits_train, _, self.transition_params = self._build_ner_head(bert_out=bert_out_train)
        #         _, self.ner_preds_inference, _ = self._build_ner_head(bert_out=bert_out_pred)
        #
        #     # re
        #     with tf.variable_scope(self.re_scope):
        #         if self.config["model"]["re"]["use_entity_emb"]:
        #             bert_dim = self.config["model"]["bert"]["dim"]
        #             self.ner_emb = tf.keras.layers.Embedding(num_labels, bert_dim)
        #             if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
        #                 self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
        #             self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])
        #
        #         if self.config["model"]["re"]["use_birnn"]:
        #             self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])
        #
        #         self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])
        #
        #         self.re_logits_train, self.num_entities = self._build_re_head(
        #             bert_out=bert_out_train, ner_labels=self.ner_labels_ph
        #         )
        #         re_logits_true_entities, _ = self._build_re_head(
        #             bert_out=bert_out_pred, ner_labels=self.ner_labels_ph
        #         )
        #         re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
        #             bert_out=bert_out_pred, ner_labels=self.ner_preds_inference
        #         )
        #
        #         self.re_labels_true_entities = tf.argmax(re_logits_true_entities, axis=-1)
        #         self.re_labels_pred_entities = tf.argmax(re_logits_pred_entities, axis=-1)
        #
        #     self._set_loss()
        #     self._set_train_op()

        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

        if self.config["model"]["ner"]["use_birnn"]:
            self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

        self.dense_ner_labels = tf.keras.layers.Dense(self.config["model"]["ner"]["num_labels"])

        self.ner_logits_train, _, self.transition_params = self._build_re_head_fn(bert_out=self.bert_out_train)
        _, self.ner_preds_inference, _ = self._build_re_head_fn(bert_out=self.bert_out_pred)

    def _build_re_head_fn(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x, num_entities = self._get_entities_representation(bert_out=bert_out, ner_labels=ner_labels)

        # encoding of pairs
        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.entity_pairs_enc(inputs=inputs, training=self.training_ph)  # [N, num_ent, num_ent, num_relation]
        return logits, num_entities

    def _get_entities_representation(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        bert_out ->
        ner_labels -> x_ner

        Выход - логиты отношений

        Пусть v_context - контекстный вектор первого токена сущности или триггера события,
              v_label - обучаемый с нуля вектор лейбла или триггера события
              v_entity - обучаемый с нуля вектор именной сущности

        Есть несколько способов векторизации сущностей и триггеров событий:

        1. v_context
        2. v_context + v_label
        3. сущнсоть - v_entity, триггер - v_context + v_label

        :param bert_out: tf.Tensor of shape [batch_size, num_pieces_max, bert_dim] and type tf.float32
        :param ner_labels: tf.Tensor of shape [batch_size, num_tokens_max] and type tf.int32
        :return:
        """
        # dropout
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x_bert = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        if self.ner_emb is not None:
            x_emb = self._get_ner_embeddings(ner_labels=ner_labels)
            x = x_bert + x_emb
        else:
            x = x_bert

        if self.birnn_re is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_re(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]

        # вывод координат первых токенов сущностей
        start_ids = tf.constant(self.config["model"]["ner"]["start_ids"], dtype=tf.int32)
        coords, num_entities = get_batched_coords_from_labels(
            labels_2d=ner_labels, values=start_ids, sequence_len=self.num_tokens_ph
        )

        # tokens -> entities
        x = tf.gather_nd(x, coords)   # [batch_size, num_entities_max, bert_bim or cell_dim * 2]
        return x, num_entities

    def _get_re_loss(self):
        no_rel_id = self.config["model"]["re"]["no_relation_id"]
        logits_shape = tf.shape(self.re_logits_train)
        labels_shape = logits_shape[:3]
        labels = get_dense_labels_from_indices(indices=self.re_labels_ph, shape=labels_shape, no_label_id=no_rel_id)
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.re_logits_train
        )  # [batch_size, num_entities, num_entities]

        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)
        mask = sequence_mask[:, None, :] * sequence_mask[:, :, None]

        masked_per_example_loss = per_example_loss * mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(mask), tf.float32)
        num_pairs = tf.maximum(num_pairs, 1.0)
        loss = total_loss / num_pairs
        loss *= self.config["model"]["re"]["loss_coef"]
        return loss
