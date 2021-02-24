import random
import os
from typing import Dict, List
from functools import partial
from itertools import chain
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer

from .utils import infer_entities_bounds
from .layers import GraphEncoder, GraphEncoderInputs
from ..data.base import Example
from ..metrics import get_ner_metrics, f1_score_micro


class BaseModel(ABC):
    """
    Interface for all models
    """

    model_scope = "model"

    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        self.loss = None
        self.train_op = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def _get_feed_dict(self, examples: List[Example], training: bool) -> Dict:
        pass

    @abstractmethod
    def _set_placeholders(self):
        pass

    @abstractmethod
    def _set_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_train_op(self):
        pass

    @abstractmethod
    def predict(self, examples: List[Example], batch_size: int = 16, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, examples: List[Example], batch_size: int = 16, **kwargs) -> Dict:
        """
        Возвращаемый словарь должен обязательно содержать ключи "score" и "loss"
        """

    def train(
            self,
            examples_train: List[Example],
            examples_eval: List[Example],
            num_epochs: int = 1,
            batch_size: int = 128,
            train_op_name: str = "train_op",
            id2label=None,
            checkpoint_path: str = None,
            scope_to_save: str = None
    ):
        train_loss = []

        # TODO: отделить конфигурацию оптимизатора от конфигурации обучения
        num_acc_steps = self.config["optimizer"]["num_accumulation_steps"]
        global_batch_size = batch_size * num_acc_steps
        epoch_steps = len(examples_train) // global_batch_size + 1
        num_train_steps = num_epochs * epoch_steps

        print(f"global batch size: {global_batch_size}")
        print(f"epoch steps: {epoch_steps}")
        print(f"num_train_steps: {num_train_steps}")

        train_op = getattr(self, train_op_name)

        epoch = 1
        best_score = -1
        num_steps_wo_improvement = 0
        num_lr_updates = 0

        if self.config["optimizer"]["reduce_lr_on_plateau"]:
            lr = self.config["optimizer"]["init_lr"]
        else:
            lr = None

        saver = None
        if checkpoint_path is not None:
            if scope_to_save is not None:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_to_save)
            else:
                var_list = tf.trainable_variables()
            saver = tf.train.Saver(var_list)

        for step in range(num_train_steps):
            examples_batch = random.sample(examples_train, batch_size)
            feed_dict, _ = self._get_feed_dict(examples_batch, training=True)
            _, loss = self.sess.run([train_op, self.loss], feed_dict=feed_dict)
            train_loss.append(loss)

            if step != 0 and step % epoch_steps == 0:

                print(f"epoch {epoch} finished. evaluation starts.")
                performance_info = self.evaluate(examples=examples_eval, batch_size=batch_size, id2label=id2label)
                score = performance_info["score"]

                if score > best_score:
                    print("new best score:", score)
                    best_score = score
                    num_steps_wo_improvement = 0

                    if saver is not None:
                        saver.save(self.sess, checkpoint_path)
                        print(f"saved new head to {checkpoint_path}")
                else:
                    num_steps_wo_improvement += 1
                    print("current score:", score)
                    print("best score:", best_score)
                    print("steps wo improvement:", num_steps_wo_improvement)

                    if num_steps_wo_improvement == self.config["optimizer"]["max_steps_wo_improvement"]:
                        print("training finished due to max number of steps wo improvement encountered.")
                        break

                    if self.config["optimizer"]["reduce_lr_on_plateau"]:
                        if num_steps_wo_improvement % self.config["optimizer"]["lr_reduce_patience"] == 0:
                            lr_old = lr
                            lr *= self.config["optimizer"]["lr_reduction_factor"]
                            num_lr_updates += 1
                            print(f"lr reduced from {lr_old} to {lr}")

                if self.config["optimizer"]['custom_schedule']:
                    lr = 1e-3
                    if epoch < 100:
                        lr = 1e-3
                    else:
                        lr = lr * 0.965 ** ((epoch - 100) + 1)

                lr = max(lr, self.config['optimizer']['min_lr'])

                print("lr:", lr)
                print("num lr updates:", num_lr_updates)
                print('=' * 50)

                epoch += 1

    def restore(self, model_dir: str):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope)
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(model_dir, "model.ckpt")
        saver.restore(self.sess, checkpoint_path)

    def initialize(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for v, flag in zip(global_vars, is_not_initialized) if not flag]
        if not_initialized_vars:
            self.sess.run(tf.variables_initializer(not_initialized_vars))
        self.sess.run(tf.tables_initializer())


class BertJointModel(BaseModel):
    """
    1. Поиск сущностей и триггеров событий (flat ner)
    2. Поиск отношений между сущностями и аргументов событий

    https://arxiv.org/abs/1812.11275
    """

    def __init__(self, sess, config):
        """
        config = {
            "model": {
                "bert": {
                    "dir": "~/bert",
                    "dim": 768,
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
                    "rnn": {
                        "num_layers": 1,
                        "cell_dim": 128,
                        "dropout": 0.5
                    }
                },
                "re": {
                    "no_relation_id": 0,
                    "rnn": {
                        "num_layers": 1,
                        "cell_dim": 128,
                        "dropout": 0.5
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
                "init_lr": 5e-5,
                "num_train_steps": 100000,
                "num_warmup_steps": 10000
            }
        }
        """
        super().__init__(sess=sess, config=config)

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
        self.num_entities_ph = None
        self.re_labels_ph = None

        # common
        self.training_ph = None

        # TENSORS
        self.loss_ner = None
        self.loss_re = None
        self.ner_preds_inference = None
        self.re_preds_inference = None

    def build(self):
        with tf.variable_scope("model"):
            bert_out_train = self._build_bert(training=True)  # [batch_size, num_pieces_max, bert_dim]
            bert_out_inference = self._build_bert(training=False)

            # common tensors
            # ner
            with tf.variable_scope("ner"):
                num_labels = self.config["model"]["ner"]["num_labels"]
                cell_dim = self.config["model"]["ner"]["rnn"]["cell_dim"]
                dropout = self.config["model"]["ner"]["rnn"]["dropout"]

                lstm = tf.keras.layers.LSTM(cell_dim, return_sequences=True, dropout=dropout)
                bilstm = tf.keras.layers.Bidirectional(lstm)  # TODO: multi-layer support; "enabled" flag
                dense_labels = tf.keras.layers.Dense(num_labels)

                ner_head_fn = partial(
                    self._build_ner_head,
                    bilstm=bilstm,
                    dense_logits=dense_labels
                )
                ner_logits_train, _, transition_params = ner_head_fn(x=bert_out_train)
                _, self.ner_preds_inference, _ = ner_head_fn(x=bert_out_inference)

            # re
            with tf.variable_scope("re"):
                bert_dim = self.config["model"]["bert"]["dim"]
                cell_dim = self.config["model"]["re"]["rnn"]["cell_dim"]
                dropout = self.config["model"]["re"]["rnn"]["dropout"]

                ner_emb = tf.keras.layers.Embedding(num_labels, bert_dim)
                lstm = tf.keras.layers.LSTM(cell_dim, return_sequences=True, dropout=dropout)
                bilstm = tf.keras.layers.Bidirectional(lstm)  # TODO: multi-layer support; "enabled" flag
                enc = GraphEncoder(
                    num_mlp_layers=self.config["model"]["re"]["biaffine"]["num_mlp_layers"],
                    head_dim=self.config["model"]["re"]["biaffine"]["head_dim"],
                    dep_dim=self.config["model"]["re"]["biaffine"]["dep_dim"],
                    output_dim=self.config["model"]["re"]["biaffine"]["num_labels"],
                    dropout=self.config["model"]["re"]["biaffine"]["dropout"],
                    activation=self.config["model"]["re"]["biaffine"]["activation"],
                )

                re_head_fn = partial(
                    self._build_re_head,
                    ner_emb=ner_emb,
                    bilstm=bilstm,
                    enc=enc,
                )
                re_logits_train = re_head_fn(bert_out=bert_out_train, ner_labels=self.ner_labels_ph)
                re_logits_inference = re_head_fn(bert_out=bert_out_inference, ner_labels=self.ner_preds_inference)

                self.re_preds_inference = tf.argmax(re_logits_inference, axis=-1)

            self._set_loss(ner_logits=ner_logits_train, transition_params=transition_params, re_logits=re_logits_train)

    def evaluate(self, examples: List[Example], batch_size: int = 16, **kwargs) -> Dict:
        """
        metrics = {
            "ner": {},
            "re": {},
            "total": {}
        }
        """
        y_true_ner = []
        y_pred_ner = []

        y_true_re = []
        y_pred_re = []

        no_rel_id = self.config["model"]["re"]["no_relation_id"]
        id_to_ner_label = kwargs["id_to_ner_label"]
        id_to_re_label = kwargs["id_to_re_label"]

        loss = 0.0
        loss_ner = 0.0
        loss_re = 0.0
        num_batches = 0

        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict = self._get_feed_dict(examples_batch, training=False)
            loss_i, loss_ner_i, loss_re_i, ner_labels_pred, rel_labels_pred = self.sess.run(
                [self.loss, self.loss_ner, self.loss_re, self.ner_preds_inference, self.re_preds_inference],
                feed_dict=feed_dict
            )
            loss += loss_i
            loss_ner += loss_ner_i
            loss_re += loss_re_i

            for i, x in enumerate(examples_batch):
                # ner
                y_true_ner.append([t.labels[0] for t in x.tokens])
                y_pred_ner.append([id_to_ner_label[j] for j in ner_labels_pred[i, :len(x.tokens)]])

                # re TODO: рассмотреть случаи num_events == 0
                num_entities = len(x.entities)
                arcs_true = np.full((num_entities, num_entities), no_rel_id, dtype=np.int32)

                for arc in x.arcs:
                    arcs_true[arc.head, arc.dep] = arc.rel

                arcs_pred = rel_labels_pred[i, :num_entities, :num_entities]
                y_true_re += [id_to_re_label[j] for j in arcs_true.flatten()]
                y_pred_re += [id_to_re_label[j] for j in arcs_pred.flatten()]

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches
        loss_ner /= num_batches
        loss_re /= num_batches

        # ner
        joiner = self.config["model"]["ner"]["prefix_joiner"]
        entity_level_metrics = get_ner_metrics(y_true=y_true_ner, y_pred=y_pred_ner, joiner=joiner)
        token_level_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, output_dict=True)

        # re
        re_micro_metrics = f1_score_micro(y_true=y_true_re, y_pred=y_pred_re, trivial_label=no_rel_id)
        label_level_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, output_dict=True)

        # total
        score = entity_level_metrics["micro"]["f1"] * 0.5 + re_micro_metrics["f1"] * 0.5

        # TODO: написать функцию, которая всё это красиво отображает (см. код classification_report)
        performance_info = {
            "ner": {
                "loss": loss_ner,
                "metrics": {
                    "entity_level": entity_level_metrics,
                    "token_level": token_level_metrics
                }
            },
            "re": {
                "loss": loss_re,
                "metrics": {
                    "micro": re_micro_metrics,
                    "label_level": label_level_metrics
                },
            },
            "loss": loss,
            "score": score
        }

        return performance_info

    # TODO: implement
    def predict(self, examples: List[Example], batch_size: int = 16, **kwargs):
        pass

    def initialize(self):
        bert_dir = ""  # TODO: брать из конфига
        var_list = {
            self._actual_name_to_checkpoint_name(x.name): x for x in tf.trainable_variables()
            if x.name.startswith(f"{self.model_scope}/bert")
        }
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
        saver.restore(self.sess, checkpoint_path)

        super().initialize()

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
        num_entities = []
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
            segment_ids.append(0)

            num_pieces_i = 0
            num_tokens_i = 0
            ner_labels_i = []
            ptr = 1

            # tokens
            for t in x.tokens:
                num_tokens_i += 1
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.pieces)
                num_pieces_i += num_pieces_ij
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                ner_labels_i += t.label_ids
                ptr += num_pieces_ij

            # relations
            for arc in x.arcs:
                assert isinstance(arc.head, int), "encode heads"
                assert isinstance(arc.dep, int), "encode deps"
                assert isinstance(arc.rel, int), "encode relations"
                re_labels.append((i, arc.head, arc.dep, arc.rel))

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids.append(0)

            # write
            num_pieces.append(num_pieces_i)
            num_tokens.append(num_tokens_i)
            input_ids.append(input_ids_i)
            input_mask.append(input_mask_i)
            segment_ids.append(segment_ids_i)
            ner_labels.append(ner_labels_i)
            first_pieces_coords.append(first_pieces_coords_i)

        # padding
        pad_token_id = self.config["model"]["bert"]["pad_token_id"]
        pad_label_id = self.config["model"]["ner"]["no_entity_id"]
        num_tokens_max = max(num_tokens)
        num_pieces_max = max(num_pieces)
        for i in range(len(examples)):
            input_ids[i] += [pad_token_id] * (num_pieces_max - num_pieces[i])
            input_mask[i] += [0] * (num_pieces_max - num_pieces[i])
            segment_ids[i] += [0] * (num_pieces_max - num_pieces[i])
            ner_labels[i] += [pad_label_id] * (num_tokens_max - num_tokens[i])
            first_pieces_coords[i] += [(i, 0)] * (num_tokens_max - num_tokens[i])

        first_pieces_coords = list(chain(*first_pieces_coords))

        d = {
            # bert
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,

            # ner
            self.first_pieces_coords_ph: first_pieces_coords,
            self.num_pieces_ph: num_pieces,
            self.num_tokens_ph: num_tokens,
            self.ner_labels_ph: ner_labels,

            # re
            self.num_entities_ph: num_entities,
            self.re_labels_ph: re_labels,

            # common
            self.training_ph: training
        }
        return d

    def _set_placeholders(self):
        # bert inputs
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        # ner inputs
        # [id_example, id_piece]
        self.first_pieces_coords_ph = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="first_pieces_coords")
        self.num_pieces_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_pieces")
        self.num_tokens_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_tokens")
        self.ner_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ner_labels")

        # re inputs
        # [id_example, id_head, id_dep, id_rel]
        self.num_entities_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_entities")
        self.re_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, 4], name="re_labels")

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.int32, shape=None, name="training_ph")

    def _set_loss(self, ner_logits, transition_params, re_logits):
        self.loss_ner = self._get_ner_loss(logits=ner_logits, transition_params=transition_params)
        self.loss_re = self._get_re_loss(logits=re_logits)
        self.loss = self.loss_ner + self.loss_re

    def _set_train_op(self):
        train_op = create_optimizer(
            loss=self.loss,
            init_lr=self.config["training"]["init_lr"],
            num_train_steps=self.config["training"]["num_train_steps"],
            num_warmup_steps=self.config["training"]["num_warmup_steps"],
            use_tpu=False
        )
        return train_op

    def _build_bert(self, training):
        bert_dir = self.config["model"]["bert"]["dir"]
        bert_scope = self.config["model"]["bert"]["scope"]
        reuse = not training
        with tf.variable_scope(bert_scope, reuse=reuse):
            bert_config = BertConfig.from_json_file(os.path.join(bert_dir, "bert_config.json"))
            model = BertModel(
                config=bert_config,
                is_training=training,
                input_ids=self.input_ids_ph,
                input_mask=self.input_mask_ph,
                token_type_ids=self.segment_ids_ph
            )
            x = model.get_sequence_output()
        return x

    def _build_ner_head(self,  x, bilstm, dense_logits):
        use_crf = self.config["model"]["ner"]["use_crf"]
        num_labels = self.config["model"]["ner"]["num_labels"]
        cell_dim = self.config["model"]["ner"]["rnn"]["cell_dim"]

        sequence_mask = tf.sequence_mask(self.num_pieces_ph)
        x = bilstm(x, training=self.training_ph, mask=sequence_mask)
        x = self._get_embeddings_by_coords(x, coords=self.first_pieces_coords_ph, d=cell_dim * 2)

        logits = dense_logits(x)

        if use_crf:
            with tf.variable_scope("crf"):
                transition_params = tf.get_variable("transition_params", [num_labels, num_labels], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, transition_params, self.num_tokens_ph)
        else:
            pred_ids = tf.argmax(logits, axis=-1)

        return logits, pred_ids, transition_params

    def _build_re_head(self, bert_out, ner_labels, ner_emb, bilstm, enc):
        if ner_emb is not None:
            x_emb = ner_emb(ner_labels)
            x = bert_out + x_emb
        else:
            x = bert_out
        sequence_mask = tf.sequence_mask(self.num_tokens_ph)
        x = bilstm(x, training=self.training_ph, mask=sequence_mask)

        start_ids = tf.constant(self.config["model"]["ner"]["start_ids"])
        coords, num_entities = infer_entities_bounds(
            label_ids=ner_labels, sequence_len=self.num_tokens_ph, bound_ids=start_ids
        )

        cell_dim = self.config["model"]["re"]["rnn"]["cell_dim"]
        x = self._get_embeddings_by_coords(x, coords=coords, d=cell_dim * 2)  # [batch_size, num_entities, d]

        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = enc(inputs=inputs, training=self.training_ph)
        return logits

    def _get_ner_loss(self, logits, transition_params):
        use_crf = self.config["model"]["ner"]["use_crf"]
        if use_crf:
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.ner_labels_ph,
                sequence_lengths=self.num_tokens_ph,
                transition_params=transition_params
            )
            loss = -tf.reduce_mean(log_likelihood)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ner_labels_ph, logits=logits)
            loss = tf.reduce_mean(loss)
        return loss

    def _get_re_loss(self, logits):
        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        logits_shape = tf.shape(logits)
        labels = tf.broadcast_to(no_rel_id, logits_shape[:3])  # [batch_size, num_entities, num_entities]
        labels = tf.tensor_scatter_nd_update(
            tensor=labels,
            indices=self.re_labels_ph[:, :-1],
            updates=self.re_labels_ph[:, -1],
        )  # [batch_size, num_entities, num_entities]
        # [batch_size, num_entities, num_entities]:
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        mask = tf.cast(tf.sequence_mask(self.num_entities_ph), tf.float32)  # [batch_size, num_entities]
        masked_per_example_loss = per_example_loss * mask[:, :, None] * mask[:, None, :]
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(self.num_entities_ph ** 2), tf.float32)
        loss = total_loss / num_pairs
        return loss

    @classmethod
    def _actual_name_to_checkpoint_name(cls, name: str) -> str:
        name = name[len(cls.model_scope) + 1:]
        name = name.replace(":0", "")
        return name

    @staticmethod
    def _get_embeddings_by_coords(x, coords, d: int):
        """
        Итоговую размерность (d) нужно задавать явно, иначе её не получится вывести для полносвязного слоя
        """
        batch_size = tf.shape(x)[0]
        x = tf.gather_nd(x, coords)  # [batch_size * num_tokens_max, d]
        x = tf.reshape(x, [batch_size, -1, d])  # [batch_size, num_tokens_max, d]
        return x


# class RelationExtractor:
#     """
#     Предполагается, что NER уже решён
#     """
#     scope = "re_head"
#
#     def __init__(self, sess, config):
#         self.sess = sess
#         self.config = config
#
#         # placeholders
#         self.tokens_ph = None
#         self.sequence_len_ph = None
#         self.num_entities_ph = None
#         self.ner_labels_ph = None
#         self.entity_start_ids_ph = None
#         self.entity_end_ids_ph = None
#         self.type_ids_ph = None
#         self.training_ph = None
#         self.lr_ph = None
#
#         # некоторые нужные тензоры
#         self.rel_labels_pred = None
#         self.loss = None
#
#         # ops
#         self.train_op = None
#         self.acc_op = None
#         self.reset_op = None
#
#         # for debug
#         self.global_step = None
#         self.all_are_finite = None
#         self.x_span = None
#
#     def build(self):
#         self._set_placeholders()
#
#         # конфиги голов
#         config_embedder = self.config["model"]["embedder"]
#         config_re = self.config["model"]["re"]
#
#         # embedder
#         # TODO: добавить возможность выбора bert
#         if config_embedder["type"] == "elmo":
#             elmo = hub.Module(config_embedder["dir"], trainable=False)
#             input_dict = {
#                 "tokens": self.tokens_ph,
#                 "sequence_len": self.sequence_len_ph
#             }
#             x = elmo(input_dict, signature="tokens", as_dict=True)["elmo"]  # [N, T, elmo_dim]
#
#             elmo_dropout = tf.keras.layers.Dropout(config_embedder["dropout"])
#             x = elmo_dropout(x, training=self.training_ph)
#         else:
#             raise NotImplementedError
#
#         # sequence_mask (нужна и в ner, и в re)
#         sequence_mask = tf.cast(tf.sequence_mask(self.sequence_len_ph), tf.float32)
#
#         with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
#
#             # эмбеддинги лейблов именных сущностей
#             if config_re["ner_embeddings"]["use"]:
#                 ner_emb = tf.keras.layers.Embedding(
#                     input_dim=config_re["ner_embeddings"]["num_labels"],
#                     output_dim=config_re["ner_embeddings"]["dim"]
#                 )(self.ner_labels_ph)
#                 ner_dropout = tf.keras.layers.Dropout(config_re["ner_embeddings"]["dropout"])
#                 ner_emb = ner_dropout(ner_emb, training=self.training_ph)
#
#                 # merged
#                 if config_re["merged_embeddings"]["merge_mode"] == "concat":
#                     x = tf.concat([x, ner_emb], axis=-1)
#                 elif config_re["merged_embeddings"]["merge_mode"] == "sum":
#                     # TODO: вставить assert на равенство размерности эмбеддингов сущностей и разметности elmo
#                     x += ner_emb
#                 else:
#                     raise NotImplementedError
#
#                 x = tf.keras.layers.Dropout(config_re["merged_embeddings"]["dropout"])(x, training=self.training_ph)
#
#                 if config_re["merged_embeddings"]["layernorm"]:
#                     x = tf.keras.layers.LayerNormalization()(x)
#
#             # обучаемые с нуля верхние слои:
#             if config_embedder["attention"]["enabled"]:
#                 x = self._stacked_attention(x, config=config_embedder["attention"],
#                                             mask=sequence_mask)  # [N, T, d_model]
#                 d_model = config_embedder["attention"]["num_heads"] * config_embedder["attention"]["head_dim"]
#             if config_embedder["rnn"]["enabled"]:
#                 x = self._stacked_rnn(x, config=config_embedder["rnn"], mask=sequence_mask)
#                 d_model = config_embedder["rnn"]["cell_dim"] * 2
#
#             # векторные представления сущностей
#             x = self._get_entity_embeddings(x, d_model=d_model)  # [N, num_entities, d_model]
#
#             # логиты отношений
#             relations_encoder = GraphEncoder(
#                 num_mlp_layers=config_re["mlp"]["num_layers"],
#                 head_dim=config_re["bilinear"]["hidden_dim"],
#                 dep_dim=config_re["bilinear"]["hidden_dim"],
#                 output_dim=config_re["bilinear"]["num_labels"],
#                 dropout=config_re["mlp"]["dropout"],
#                 activation=config_re["mlp"]["activation"]
#             )
#             inputs = GraphEncoderInputs(head=x, dep=x)
#             logits = relations_encoder(inputs=inputs, training=self.training_ph)  # [N, num_heads, num_deps, num_relations]
#             self.rel_labels_pred = tf.argmax(logits, axis=-1)  # [N, num_entities, num_entities]
#
#             # логиты кореференций
#
#         self._set_loss(logits=logits)
#         self._set_train_op()
#         self.sess.run(tf.global_variables_initializer())
#
#     def train(
#             self,
#             train_examples,
#             eval_examples,
#             no_rel_id: int,
#             num_epochs=1,
#             batch_size=128,
#             plot_step=10,
#             plot_train_steps=1000,
#             train_op_name="train_op",
#             id2label=None,
#             checkpoint_path=None
#     ):
#         train_loss = []
#         eval_loss = []
#
#         eval_las = []
#         eval_uas = []
#         clf_reports = []
#
#         def plot():
#             clear_output()
#
#             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
#
#             ax1.set_title("train loss")
#             ax1.plot(train_loss[-plot_train_steps:], label="loss")
#             ax1.grid()
#             ax1.legend()
#
#             ax2.set_title("eval loss")
#             ax2.plot(eval_loss, marker='o', label="total loss")
#             ax2.grid()
#             ax2.legend()
#
#             ax3.set_title("f1")
#             ax3.plot(eval_las, marker='o', label='right triple (a, b, r)')
#             ax3.plot(eval_uas, marker='o', label='right pair (a, b)')
#             ax3.legend()
#             ax3.grid()
#
#             plt.show()
#
#         def print_clf_report(y_true, y_pred):
#             if id2label is not None:
#                 target_names = [id2label[i] for i in sorted(set(y_true) | set(y_pred))]
#             else:
#                 target_names = None
#             print(classification_report(y_true, y_pred, target_names=target_names))
#
#         # TODO: отделить конфигурацию оптимизатора от конфигурации обучения
#         num_acc_steps = self.config["optimizer"]["num_accumulation_steps"]
#         global_batch_size = batch_size * num_acc_steps
#         epoch_steps = len(train_examples) // global_batch_size + 1
#         num_train_steps = num_epochs * epoch_steps
#
#         print(f"global batch size: {global_batch_size}")
#         print(f"epoch steps: {epoch_steps}")
#         print(f"num_train_steps: {num_train_steps}")
#
#         train_op = getattr(self, train_op_name)
#
#         epoch = 1
#         best_score = -1
#         num_steps_wo_improvement = 0
#         num_lr_updates = 0
#
#         if self.config["optimizer"]["reduce_lr_on_plateau"]:
#             lr = self.config["optimizer"]["init_lr"]
#         else:
#             lr = None
#
#         saver = None
#         if checkpoint_path is not None:
#             var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
#             saver = tf.train.Saver(var_list)
#
#         for step in range(num_train_steps):
#             examples_batch = random.sample(train_examples, batch_size)
#             feed_dict, _ = self._get_feed_dict(examples_batch, training=True)
#             if self.config["optimizer"]["reduce_lr_on_plateau"]:
#                 feed_dict[self.lr_ph] = lr
#             _, loss = self.sess.run([train_op, self.loss], feed_dict=feed_dict)
#             train_loss.append(loss)
#             # print(f"loss: {loss}")
#
#             # if step % plot_step == 0:
#             #     plot()
#
#             if step != 0 and step % epoch_steps == 0:
#                 print(f"epoch {epoch} finished. evaluation starts.")
#                 losses_tmp = []
#
#                 y_true_arcs_types = []
#                 y_pred_arcs_types = []
#
#                 for start in range(0, len(eval_examples), batch_size):
#                     end = start + batch_size
#                     examples_batch = eval_examples[start:end]
#                     feed_dict, id2index = self._get_feed_dict(examples_batch, training=False)
#                     loss, rel_labels_pred = self.sess.run([self.loss, self.rel_labels_pred], feed_dict=feed_dict)
#                     losses_tmp.append(loss)
#
#                     for i, x in enumerate(examples_batch):
#
#                         arcs_true = np.full((x.num_entities, x.num_entities), no_rel_id, dtype=np.int32)
#
#                         for arc in x.arcs:
#                             id_head = id2index[(x.id, arc.head)]
#                             id_dep = id2index[(x.id, arc.dep)]
#                             arcs_true[id_head, id_dep] = arc.rel
#
#                         arcs_pred = rel_labels_pred[i, :x.num_entities, :x.num_entities]
#
#                         y_true_arcs_types.append(arcs_true.flatten())
#                         y_pred_arcs_types.append(arcs_pred.flatten())
#
#                 y_true_arcs_types = np.concatenate(y_true_arcs_types)
#                 y_pred_arcs_types = np.concatenate(y_pred_arcs_types)
#
#                 print_clf_report(y_true_arcs_types, y_pred_arcs_types)
#                 re_metrics = compute_f1(labels=y_true_arcs_types, preds=y_pred_arcs_types)
#
#                 print('evaluation results:')
#                 print(re_metrics)
#
#                 score = re_metrics['f1']
#
#                 if score > best_score:
#                     print("new best score:", score)
#                     best_score = score
#                     num_steps_wo_improvement = 0
#
#                     if saver is not None:
#                         saver.save(self.sess, checkpoint_path)
#                         print(f"saved new head to {checkpoint_path}")
#                 else:
#                     num_steps_wo_improvement += 1
#                     print("current score:", score)
#                     print("best score:", best_score)
#                     print("steps wo improvement:", num_steps_wo_improvement)
#
#                     if num_steps_wo_improvement == self.config["optimizer"]["max_steps_wo_improvement"]:
#                         print("training finished due to max number of steps wo improvement encountered.")
#                         break
#
#                     if self.config["optimizer"]["reduce_lr_on_plateau"]:
#                         if num_steps_wo_improvement % self.config["optimizer"]["lr_reduce_patience"] == 0:
#                             lr_old = lr
#                             lr *= self.config["optimizer"]["lr_reduction_factor"]
#                             num_lr_updates += 1
#                             print(f"lr reduced from {lr_old} to {lr}")
#
#                 if self.config["optimizer"]['custom_schedule']:
#                     lr = 1e-3
#                     if epoch < 100:
#                         lr = 1e-3
#                     else:
#                         lr = lr * 0.965 ** ((epoch - 100) + 1)
#
#                 lr = max(lr, self.config['optimizer']['min_lr'])
#
#                 print("lr:", lr)
#                 print("num lr updates:", num_lr_updates)
#                 print('=' * 50)
#
#                 epoch += 1
#
#                 # eval_loss.append(np.mean(losses_tmp))
#
#                 # eval_las.append(re_metrics.f1_arcs_types)
#                 # eval_uas.append(re_metrics.f1_arcs)
#
#                 # plot()
#         # plot()
#         return clf_reports
#
#     def predict(self, examples, batch_size=128):
#         filename2id_arc = defaultdict(int)
#         for start in range(0, len(examples), batch_size):
#             end = start + batch_size
#             examples_batch = examples[start:end]
#             feed_dict, id2index = self._get_feed_dict(examples_batch, training=False)
#             rel_labels_pred = self.sess.run(self.rel_labels_pred, feed_dict=feed_dict)
#             d = {(id_example, i): id_entity for (id_example, id_entity), i in id2index.items()}
#             assert len(d) == len(id2index)
#             for i, x in enumerate(examples_batch):
#                 for j in range(x.num_entities):
#                     for k in range(x.num_entities):
#                         id_rel = rel_labels_pred[i, j, k]
#                         if id_rel != 0:
#                             id_head = d[(x.id, j)]
#                             id_dep = d[(x.id, k)]
#                             id_arc = filename2id_arc[x.filename]
#                             arc = Arc(id=id_arc, head=id_head, dep=id_dep, rel=id_rel)
#                             x.arcs.append(arc)
#                             filename2id_arc[x.filename] += 1
#
#     def restore(self, model_dir):
#         var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
#         saver = tf.train.Saver(var_list)
#         checkpoint_path = os.path.join(model_dir, "model.ckpt")
#         saver.restore(self.sess, checkpoint_path)
#
#     def initialize(self):
#         global_vars = tf.global_variables()
#         is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
#         not_initialized_vars = [v for v, flag in zip(global_vars, is_not_initialized) if not flag]
#         if not_initialized_vars:
#             self.sess.run(tf.variables_initializer(not_initialized_vars))
#         self.sess.run(tf.tables_initializer())
#
#     def _get_feed_dict(self, examples, training):
#         # tokens
#         pad = "[PAD]"
#         tokens = [x.tokens for x in examples]
#         sequence_len = [x.num_tokens for x in examples]
#         num_tokens_max = max(sequence_len)
#         tokens = [x + [pad] * (num_tokens_max - l) for x, l in zip(tokens, sequence_len)]
#
#         # ner labels
#         other_label_id = self.config["model"]["re"]["ner_other_label_id"]
#         ner_labels = [x.labels + [other_label_id] * (num_tokens_max - l) for x, l in zip(examples, sequence_len)]
#
#         # entities
#         num_entities = [x.num_entities for x in examples]
#         if training:
#             assert sum(num_entities) > 0, "it will not be impossible to compute loss due to the absence of entities"
#         num_entities_max = max(num_entities)
#         entity_start_ids = []
#         entity_end_ids = []
#         id2index = {}
#         # не 0, т.к. при выводе векторного представления спана (i, j) используется
#         # в том числе вектор токена i - 1. на нулевой позиции находится специальный
#         # токен начала последовтаельности.
#         pad_start = pad_end = 1
#         for i, x in enumerate(examples):
#             assert x.id is not None
#             for j, entity in enumerate(x.entities):
#                 assert entity.id is not None
#                 id2index[(x.id, entity.id)] = j
#                 entity_start_ids.append((i, entity.start_token_id))
#                 entity_end_ids.append((i, entity.end_token_id))
#             for _ in range(num_entities_max - x.num_entities):
#                 entity_start_ids.append((i, pad_start))
#                 entity_end_ids.append((i, pad_end))
#
#         # arcs
#         type_ids = []
#         for i, x in enumerate(examples):
#             for arc in x.arcs:
#                 id_head = id2index[(x.id, arc.head)]
#                 id_dep = id2index[(x.id, arc.dep)]
#                 type_ids.append((i, id_head, id_dep, arc.rel))
#         # если в батче нет ни одного отношения, то не получится посчитать лосс.
#         # решение - добавить одно отношение с лейблом NO_RELATION
#         if len(type_ids) == 0:
#             for i, x in enumerate(examples):
#                 if x.num_entities > 0:
#                     type_ids.append((i, 0, 0, self.config['model']['re']['no_rel_id']))
#                     break
#
#         # feed_dict
#         feed_dict = {
#             self.tokens_ph: tokens,
#             self.sequence_len_ph: sequence_len,
#             self.ner_labels_ph: ner_labels,
#             self.num_entities_ph: num_entities,
#             self.entity_start_ids_ph: entity_start_ids,
#             self.entity_end_ids_ph: entity_end_ids,
#             self.type_ids_ph: type_ids,
#             self.training_ph: training
#         }
#
#         return feed_dict, id2index
#
#     def _set_placeholders(self):
#         # для elmo
#         self.tokens_ph = tf.placeholder(tf.string, shape=[None, None], name="tokens_ph")
#         self.sequence_len_ph = tf.placeholder(tf.int32, shape=[None], name="sequence_len_ph")
#
#         # для эмбеддингов сущнсотей
#         self.ner_labels_ph = tf.placeholder(tf.int32, shape=[None, None], name="ner_ph")
#
#         # для маскирования на уровне сущностей
#         self.num_entities_ph = tf.placeholder(tf.int32, shape=[None], name="num_entities_ph")
#
#         # для вывода эмбеддингов спанов сущнсотей; [id_example, start]
#         self.entity_start_ids_ph = tf.placeholder(tf.int32, shape=[None, 2], name="entity_start_ids_ph")
#         self.entity_end_ids_ph = tf.placeholder(tf.int32, shape=[None, 2], name="entity_end_ids_ph")
#
#         # для обучения re; [id_example, id_head, id_dep, id_rel]
#         self.type_ids_ph = tf.placeholder(tf.int32, shape=[None, 4], name="type_ids_ph")
#
#         # для включения / выключения дропаутов
#         self.training_ph = tf.placeholder(tf.bool, shape=None, name="training_ph")
#
#         self.lr_ph = tf.placeholder(tf.float32, shape=None, name="lr_ph")
#
#     def _set_loss(self, logits):
#         """
#         logits - tf.Tensor of shape [batch_size, num_entities, num_entities, num_relations]
#         """
#         logits_shape = tf.shape(logits)  # [4]
#         labels = tf.broadcast_to(self.config['model']['re']['no_rel_id'],
#                                  logits_shape[:3])  # [batch_size, num_entities, num_entities]
#         labels = tf.tensor_scatter_nd_update(
#             tensor=labels,
#             indices=self.type_ids_ph[:, :-1],
#             updates=self.type_ids_ph[:, -1],
#         )  # [batch_size, num_entities, num_entities]
#         per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
#                                                                           logits=logits)  # [batch_size, num_entities, num_entities]
#         mask = tf.cast(tf.sequence_mask(self.num_entities_ph), tf.float32)  # [batch_size, num_entities]
#         masked_per_example_loss = per_example_loss * mask[:, :, None] * mask[:, None, :]
#         total_loss = tf.reduce_sum(masked_per_example_loss)
#         num_pairs = tf.cast(tf.reduce_sum(self.num_entities_ph ** 2), tf.float32)
#         self.loss = total_loss / num_pairs
#
#     def _set_train_op(self):
#         tvars = tf.trainable_variables()
#         global_step = tf.train.get_or_create_global_step()
#         if self.config['optimizer']['reduce_lr_on_plateau']:
#             lr = self.lr_ph
#         else:
#             if self.config['optimizer']['noam_scheme']:
#                 lr = noam_scheme(
#                     init_lr=self.config["optimizer"]["init_lr"],
#                     global_step=global_step,
#                     warmup_steps=self.config["optimizer"]["warmup_steps"]
#                 )
#             else:
#                 lr = self.config['optimizer']['init_lr']
#         optimizer = tf.train.AdamOptimizer(lr)
#         grads = tf.gradients(self.loss, tvars)
#         if self.config["optimizer"]["clip_grads"]:
#             grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config["optimizer"]["clip_norm"])
#         self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
#
#     def _get_entity_embeddings(self, x, d_model):
#         """
#         :arg
#         x: tf.Tensor of shape [batch_size, num_tokens, d_model]
#         :return
#         x_span: tf.Tensor of shape [batch_size, num_entities, d_model]
#         """
#         config_re_span_emb = self.config["model"]["re"]["span_embeddings"]
#
#         batch_size = tf.shape(x)[0]
#         emb_type = config_re_span_emb["type"]
#
#         if emb_type == 0:
#             x_span = tf.gather_nd(x, self.entity_start_ids_ph)  # [N * num_entities, D]
#         elif emb_type == 1:
#             one = tf.tile([[0, 1]], [tf.shape(self.entity_start_ids_ph)[0], 1])
#             x_i = tf.gather_nd(x, self.entity_start_ids_ph)  # [N * num_entities, D]
#             x_i_minus_one = tf.gather_nd(x, self.entity_start_ids_ph - one)  # [N * num_entities, D]
#             x_j = tf.gather_nd(x, self.entity_end_ids_ph)  # [N * num_entities, D]
#             x_j_plus_one = tf.gather_nd(x, self.entity_end_ids_ph + one)  # [N * num_entities, D]
#
#             d_model_half = d_model // 2
#             x_start = x_j - x_i_minus_one
#             x_start = x_start[..., :d_model_half]
#             x_end = x_i - x_j_plus_one
#             x_end = x_end[..., d_model_half:]
#
#             x_span = tf.concat([x_start, x_end], axis=-1)  # [N * num_entities, D]
#             self.x_span = x_span
#         else:
#             raise ValueError(f"expected span_emb type in {{0, 1}}, got {emb_type}")
#
#         x_span = tf.reshape(x_span, [batch_size, -1, d_model])  # [N, num_entities, D]
#
#         return x_span
#
#     def _stacked_attention(self, x, config, mask):
#         d_model = config["num_heads"] * config["head_dim"]
#         x = tf.keras.layers.Dense(d_model)(x)
#         for i in range(config["num_layers"]):
#             attn = DotProductAttention(**config)
#             x = attn(x, training=self.training_ph, mask=mask)
#         return x
#
#     def _stacked_rnn(self, x, config, mask):
#         for i in range(config['num_layers']):
#             with tf.variable_scope(f"recurrent_layer_{i}"):
#                 xi = self._bidirectional(x=x, config=config, mask=mask)
#                 if config['skip_connections']:
#                     if i == 0:
#                         x = xi
#                     else:
#                         x += xi
#                 else:
#                     x = xi
#         return x
#
#     def _bidirectional(self, x, config, mask):
#         cell_name = config["cell_name"]
#         cell_dim = config["cell_dim"]
#         dropout = config["dropout"]
#         recurrent_dropout = config["recurrent_dropout"]
#
#         if cell_name == "lstm":
#             recurrent_layer = tf.keras.layers.LSTM
#         elif cell_name == "gru":
#             recurrent_layer = tf.keras.layers.GRU
#         else:
#             raise Exception(f"expected cell_name in {{lstm, gru}}, got {cell_name}")
#
#         recurrent_layer = recurrent_layer(
#             units=cell_dim,
#             dropout=dropout,
#             recurrent_dropout=recurrent_dropout,
#             return_sequences=True,
#             name=cell_name
#         )
#         bidirectional_layer = tf.keras.layers.Bidirectional(recurrent_layer, name="bidirectional")
#         x = bidirectional_layer(x, mask=mask, training=self.training_ph)
#         return x
