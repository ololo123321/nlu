import random
import os
from typing import Dict, List
from functools import partial
from abc import ABC, abstractmethod

import tensorflow as tf
from sklearn.metrics import classification_report

from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer

from .utils import infer_start_coords
from .layers import GraphEncoder, GraphEncoderInputs
from ..data.base import Example


class BaseModel(ABC):
    """
    Interface for all models
    """

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
    def predict(self, examples: List[Example], batch_size: int = 16):
        pass

    @abstractmethod
    def evaluate(self, examples: List[Example], batch_size: int = 16) -> float:
        pass

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

        def print_clf_report(y_true, y_pred):
            if id2label is not None:
                target_names = [id2label[i] for i in sorted(set(y_true) | set(y_pred))]
            else:
                target_names = None
            print(classification_report(y_true, y_pred, target_names=target_names))

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
                score = self.evaluate(examples=examples_eval, batch_size=batch_size)

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

    def restore(self, model_dir: str, scope: str):  # TODO: scope вынести в config
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
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
        super().__init__(sess=sess, config=config)

        # placeholders
        self.input_ids_ph = None
        self.input_mask_ph = None
        self.segment_ids_ph = None

        self.entity_start_ids = None
        self.start_mask_ph = None
        self.token_start_mask_ph = None
        self.labels_ph = None
        self.sequence_len_ph = None
        self.training_ph = None
        self.type_ids_ph = None
        self.num_entities_ph = None

    def build(self):
        with tf.variable_scope("model"):
            bert_out_train = self._build_bert(training=True)
            bert_out_inference = self._build_bert(training=False)

            # common tensors
            sequence_mask = tf.sequence_mask(self.sequence_len_ph)

            # ner
            with tf.variable_scope("ner"):
                num_labels = self.config["model"]["ner"]["num_labels"]
                cell_dim = self.config["model"]["ner"]["rnn"]["cell_dim"]
                dropout = self.config["model"]["ner"]["rnn"]["dropout"]
                lstm = tf.keras.layers.LSTM(cell_dim, return_sequences=True, dropout=dropout)
                bilstm = tf.keras.layers.Bidirectional(lstm)
                # TODO: all pieces -> first pieces
                dense_labels = tf.keras.layers.Dense(num_labels)
                ner_head_fn = partial(
                    self._build_ner_head,
                    sequence_mask=sequence_mask,
                    lstm=bilstm,
                    dense_logits=dense_labels
                )
                ner_logits_train, _, transition_params = ner_head_fn(x=bert_out_train)
                _, pred_ids_inference, _ = ner_head_fn(x=bert_out_inference)

            # re
            with tf.variable_scope("re"):
                bert_dim = self.config["model"]["bert"]["dim"]
                ner_emb = tf.keras.layers.Embedding(num_labels, bert_dim)
                lstm = tf.keras.layers.LSTM(cell_dim, return_sequences=True, dropout=dropout)
                bilstm = tf.keras.layers.Bidirectional(lstm)
                enc = GraphEncoder(
                    num_mlp_layers=self.config["model"]["re"]["num_mlp_layers"],
                    head_dim=self.config["model"]["re"]["head_dim"],
                    dep_dim=self.config["model"]["re"]["dep_dim"],
                    output_dim=self.config["model"]["re"]["num_labels"],
                    dropout=self.config["model"]["re"]["dropout"],
                    activation=self.config["model"]["re"]["activation"],
                )
                re_head_fn = partial(
                    self._build_re_head,
                    ner_emb=ner_emb,
                    lstm=bilstm,
                    enc=enc,
                    sequence_mask=sequence_mask
                )
                re_logits_train = re_head_fn(bert_out=bert_out_train, ner_labels=self.labels_ph)
                re_logits_inference = re_head_fn(bert_out=bert_out_inference, ner_labels=pred_ids_inference)

            self._set_loss(ner_logits=ner_logits_train, transition_params=transition_params, re_logits=re_logits_train)

        # return outputs_train, outputs_inference

    # TODO: implement
    def evaluate(self, examples, batch_size=16) -> float:
        pass

    # TODO: implement
    def predict(self, examples, batch_size=16):
        pass

    # TODO: implement
    def _get_feed_dict(self, examples, training: bool) -> Dict:
        pass

    def _set_placeholders(self):
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        self.entity_start_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="entity_start_ids")
        self.start_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="start_mask")
        self.token_start_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_start_mask")
        self.labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="labels")
        self.sequence_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_len_ph")
        self.training_ph = tf.placeholder(dtype=tf.int32, shape=None, name="training_ph")

    def _set_loss(self, ner_logits, transition_params, re_logits):
        ner_loss = self._get_ner_loss(logits=ner_logits, transition_params=transition_params)
        re_loss = self._get_re_loss(logits=re_logits)
        self.loss = ner_loss + re_loss

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
        bert_dir = ""  # TODO: брать из конфига
        bert_scope = ""  # TODO: брать из конфига
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

    def _build_ner_head(self,  x, sequence_mask, lstm, dense_logits):
        use_crf = self.config["model"]["ner"]["use_crf"]
        bert_dim = self.config["model"]["bert"]["dim"]
        num_labels = self.config["model"]["ner"]["num_labels"]

        x = lstm(x, training=self.training_ph, mask=sequence_mask)

        x_shape = tf.shape(x)
        coords, num_tokens = infer_start_coords(self.start_mask_ph)
        x = tf.gather_nd(x, coords)  # [batch_size * num_tokens_max, d]
        x = tf.reshape(x, [x_shape[0], -1, bert_dim])  # [batch_size, num_tokens_max, d]

        logits = dense_logits(x)

        if use_crf:
            with tf.variable_scope("crf"):
                transition_params = tf.get_variable("transition_params", [num_labels, num_labels], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, transition_params, num_tokens)
        else:
            pred_ids = tf.argmax(logits, axis=-1)

        return logits, pred_ids, transition_params

    def _build_re_head(self, bert_out, ner_labels, ner_emb, lstm, enc, sequence_mask):
        if ner_emb is not None:
            x_emb = ner_emb(ner_labels)
            x = bert_out + x_emb
        else:
            x = bert_out
        x = lstm(x, training=self.training_ph, mask=sequence_mask)

        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = enc(inputs=inputs, training=self.training_ph)
        return logits

    def _get_ner_loss(self, logits, transition_params):
        use_crf = self.config["model"]["ner"]["use_crf"]
        if use_crf:
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels_ph,
                sequence_lengths=self.sequence_len_ph,
                transition_params=transition_params
            )
            loss = -tf.reduce_mean(log_likelihood)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_ph, logits=logits)
            loss = tf.reduce_mean(loss)
        return loss

    def _get_re_loss(self, logits):
        logits_shape = tf.shape(logits)
        labels = tf.broadcast_to(self.config['model']['re']['no_rel_id'],
                                 logits_shape[:3])  # [batch_size, num_entities, num_entities]
        labels = tf.tensor_scatter_nd_update(
            tensor=labels,
            indices=self.type_ids_ph[:, :-1],
            updates=self.type_ids_ph[:, -1],
        )  # [batch_size, num_entities, num_entities]
        # [batch_size, num_entities, num_entities]:
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        mask = tf.cast(tf.sequence_mask(self.num_entities_ph), tf.float32)  # [batch_size, num_entities]
        masked_per_example_loss = per_example_loss * mask[:, :, None] * mask[:, None, :]
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(self.num_entities_ph ** 2), tf.float32)
        loss = total_loss / num_pairs
        return loss

    # TODO: юзать при initialize
    # def actual_name_to_checkpoint_name(name):
    #     name = name[len(MODEL_SCOPE) + 1:]
    #     name = name.replace(":0", "")
    #     return name


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
