import random
from tqdm import trange

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics import classification_report

from IPython.display import clear_output
from matplotlib import pyplot as plt

from .utils import compute_re_metrics, infer_entities_bounds, noam_scheme
from .layers import DotProductAttention, REHead


# model

class RelationExtractor:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        self.tokens_ph = None
        self.sequence_len_ph = None
        self.num_entities_ph = None
        self.ner_labels_ph = None
        self.entity_starts = None
        self.entity_ends = None
        self.type_ids_ph = None
        self.training_ph = None

        # некоторые нужные тензоры
        self.s_type_train = None
        self.s_type_inference = None
        self.transition_params = None
        self.ner_labels_pred = None
        self.loss_ner = None
        self.loss_re_train = None
        self.loss_re_inference = None
        self.loss_train = None
        self.loss_inference = None

        # ops
        self.train_op = None
        self.acc_op = None
        self.reset_op = None

        # for debug
        self.global_step = None
        self.all_are_finite = None
        self.x_span = None

    def build(self):
        self._set_placeholders()

        # конфиги голов
        config_embedder = self.config["model"]["embedder"]
        config_ner = self.config["model"]["ner"]
        config_re = self.config["model"]["re"]

        # embedder
        if config_embedder["type"] == "elmo":
            elmo = hub.Module(config_embedder["dir"], trainable=False)
            input_dict = {
                "tokens": self.tokens_ph,
                "sequence_len": self.sequence_len_ph
            }
            x = elmo(input_dict, signature="tokens", as_dict=True)["elmo"]

            elmo_dropout = tf.keras.layers.Dropout(config_embedder["dropout"])
            x = elmo_dropout(x, training=self.training_ph)
        else:
            raise NotImplementedError

        # sequence_mask (нужна и в ner, и в re)
        sequence_mask = tf.sequence_mask(self.sequence_len_ph)
        sequence_mask_float = tf.cast(sequence_mask, tf.float32)

        x = self._stacked_attention(x, config=config_embedder["attention"], mask=sequence_mask_float)

        # ner
        num_ner_labels = config_ner["num_labels"]
        with tf.variable_scope("ner_head"):
            # обучаемые с нуля верхние слои
            # x_ner = self._stacked_attention(x, config=config_ner["attention"], mask=sequence_mask_float)
            # x_ner = tf.keras.layers.Dense(num_ner_labels)(x_ner)
            x_ner = tf.keras.layers.Dense(num_ner_labels)(x)

            if config_ner["use_crf"]:
                with tf.variable_scope("transition_params"):
                    self.transition_params = tf.get_variable("transition_params",
                                                             [num_ner_labels, num_ner_labels], dtype=tf.float32)

                # ner_labels_pred: [N, T]
                self.ner_labels_pred, _ = tf.contrib.crf.crf_decode(x_ner, self.transition_params, self.sequence_len_ph)
            else:
                self.ner_labels_pred = tf.argmax(x_ner, axis=-1)

            self.ner_labels_pred = tf.stop_gradient(self.ner_labels_pred)

        # re

        def add_re_head(self, x, ner_label_ids):
            """
            при обучении берём известные истинные лейблы, при инференсе - предсказанные моделью
            """
            # при инференсе в качестве паддинга может быть использован любой лейбл
            ner_label_ids = tf.where(sequence_mask, ner_label_ids,
                                     tf.broadcast_to(config_ner["other_label_id"], tf.shape(sequence_mask)))

            with tf.variable_scope("re_head", reuse=tf.AUTO_REUSE):

                # эмбеддинги лейблов именных сущностей
                if config_re["ner_embeddings"]["use"]:
                    ner_emb = tf.keras.layers.Embedding(
                        input_dim=num_ner_labels,
                        output_dim=config_re["ner_embeddings"]["dim"]
                    )(ner_label_ids)
                    ner_dropout = tf.keras.layers.Dropout(config_re["ner_embeddings"]["dropout"])
                    ner_emb = ner_dropout(ner_emb, training=self.training_ph)

                    # merged
                    if config_re["merged_embeddings"]["merge_mode"] == "concat":
                        x = tf.concat([x, ner_emb], axis=-1)
                    elif config_re["merged_embeddings"]["merge_mode"] == "sum":
                        x += ner_emb
                    else:
                        raise NotImplementedError

                    x = tf.keras.layers.Dropout(config_re["merged_embeddings"]["dropout"])(x, training=self.training_ph)

                    if config_re["merged_embeddings"]["layernorm"]:
                        x = tf.keras.layers.LayerNormalization()(x)

                # обучаемые с нуля верхние слои:
                # attn_config = config_re["attention"]
                # x = self._stacked_attention(x, config=attn_config, mask=sequence_mask_float)

                # векторные представления сущностей

                # d_model = attn_config["num_heads"] * attn_config["head_dim"]
                # d_model = config_embedder["dim"]
                d_model = config_embedder["attention"]["num_heads"] * config_embedder["attention"]["head_dim"]
                x = self._get_entity_embeddings(x, label_ids=ner_label_ids,
                                                d_model=d_model)  # [N, num_entities, d_model]

                # билинейный слой, учащий совмесное распределение сущностей
                # TODO: сделать по-нормальному
                parser_config = {
                    "mlp": config_re["mlp"],
                    "type": config_re["bilinear"]
                }
                parser = REHead(parser_config)
                x = parser(x, training=self.training_ph)
                return x

        self.s_type_train = add_re_head(self, x=x, ner_label_ids=self.ner_labels_ph)
        self.s_type_inference = add_re_head(self, x=x, ner_label_ids=self.ner_labels_pred)

        # losses
        self.loss_ner = self._get_ner_loss(x_ner)

        self.loss_re_train = self._get_re_loss(self.s_type_train)
        self.loss_re_inference = self._get_re_loss(self.s_type_inference)

        self.loss_train = self.loss_ner + self.loss_re_train
        self.loss_inference = self.loss_ner + self.loss_re_inference

        self._set_train_op()
        self.sess.run(tf.global_variables_initializer())

    def train(self, train_examples, eval_examples, no_rel_id: int, num_epochs=1, batch_size=128, plot_step=10,
              plot_train_steps=1000):
        train_loss = []
        train_loss_ner = []
        train_loss_re = []

        eval_loss = []
        eval_loss_ner = []
        eval_loss_re = []

        eval_las = []
        eval_uas = []
        clf_reports = []

        def plot():
            clear_output()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

            ax1.set_title("train loss")
            ax1.plot(train_loss[-plot_train_steps:], label="total loss")
            ax1.plot(train_loss_ner[-plot_train_steps:], label="ner loss")
            ax1.plot(train_loss_re[-plot_train_steps:], label="re loss")
            ax1.grid()
            ax1.legend()

            ax2.set_title("eval loss")
            ax2.plot(eval_loss, marker='o', label="total loss")
            ax2.plot(eval_loss_ner, marker='o', label="ner loss")
            ax2.plot(eval_loss_re, marker='o', label="re loss")
            ax2.grid()
            ax2.legend()

            ax3.set_title("greedy attachment scores")
            ax3.plot(eval_las, marker='o', label='las')
            ax3.plot(eval_uas, marker='o', label='uas')
            ax3.legend()
            ax3.grid()

            plt.show()

        # TODO: отделить конфигурацию оптимизатора от конфигурации обучения
        num_acc_steps = self.config["optimizer"]["num_accumulation_steps"]
        global_batch_size = batch_size * num_acc_steps
        epoch_steps = len(train_examples) // global_batch_size + 1
        num_train_steps = num_epochs * epoch_steps

        print(f"global batch size: {global_batch_size}")
        print(f"epoch steps: {epoch_steps}")
        print(f"num_train_steps: {num_train_steps}")

        for step in range(num_train_steps):
            if self.config["optimizer"]["num_accumulation_steps"] == 1:
                examples_batch = random.sample(train_examples, batch_size)
                feed_dict = self._get_feed_dict(examples_batch, training=True)
                _, loss, loss_ner, loss_re_train = self.sess.run([
                    self.train_op,
                    self.loss_train,
                    self.loss_ner,
                    self.loss_re_train
                ], feed_dict=feed_dict)
                train_loss.append(loss)
                train_loss_ner.append(loss_ner)
                train_loss_re.append(loss_re_train)
                print(f"loss: {loss}; loss ner: {loss_ner}; loss re: {loss_re_train}")
            else:
                # обнуление переменных, хранящих накопленные градиенты
                # TODO: актуализировать
                self.sess.run(self.reset_op)
                losses_tmp = []
                aaf = True

                # накопление градиентов
                for _ in range(num_acc_steps):
                    examples_batch = random.sample(train_examples, batch_size)
                    feed_dict = self._get_feed_dict(examples_batch, training=True)
                    _, loss, gs, aaf_step = self.sess.run(
                        [self.acc_op, self.loss_train, self.global_step, self.all_are_finite], feed_dict=feed_dict)
                    print(gs, loss, aaf_step)
                    # with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    #     v = tf.get_variable("dependency_parser/dense/kernel/accum").eval(session=self.sess)
                    #     print("accum:")
                    #     print(v)
                    losses_tmp.append(loss)
                    aaf &= aaf_step

                # проверка финитности градиентов
                if not aaf:
                    while True:
                        examples_batch = random.sample(train_examples, batch_size)
                        feed_dict = self._get_feed_dict(examples_batch, training=True)
                        _, loss, gs, aaf_step = self.sess.run(
                            [self.acc_op, self.loss_train, self.global_step, self.all_are_finite], feed_dict=feed_dict)
                        if aaf_step:
                            break

                # обновление весов
                self.sess.run(self.train_op)
                train_loss.append(np.mean(losses_tmp))

            if step % plot_step == 0:
                plot()

            if step != 0 and step % epoch_steps == 0:
                losses_tmp = []
                losses_tmp_ner = []
                losses_tmp_re = []

                y_true_arcs_types = []
                y_pred_arcs_types = []

                for start in range(0, len(eval_examples), batch_size):
                    end = start + batch_size
                    examples_batch = eval_examples[start:end]
                    feed_dict = self._get_feed_dict(examples_batch, training=False)
                    loss, loss_ner, loss_re, s_type = self.sess.run([
                        self.loss_train,
                        self.loss_ner,
                        self.loss_re_train,
                        self.s_type_train
                    ], feed_dict=feed_dict)
                    losses_tmp.append(loss)
                    losses_tmp_ner.append(loss_ner)
                    losses_tmp_re.append(loss_re)

                    # TODO: сделать векторизовано
                    s_type_argmax = s_type.argmax(-1)  # [N, num_entities, num_entities]

                    for i, x in enumerate(examples_batch):
                        arcs_true = np.zeros((x.num_entities, x.num_entities), dtype=np.int32) + no_rel_id
                        for head_true, dep_true, rel_true in x.arcs:
                            arcs_true[head_true, dep_true] = rel_true

                        arcs_pred = s_type_argmax[i, :x.num_entities, :x.num_entities]

                        y_true_arcs_types.append(arcs_true.flatten())
                        y_pred_arcs_types.append(arcs_pred.flatten())

                y_true_arcs_types = np.concatenate(y_true_arcs_types)
                y_pred_arcs_types = np.concatenate(y_pred_arcs_types)

                clf_report = classification_report(y_true_arcs_types, y_pred_arcs_types)
                clf_reports.append(clf_report)
                print(clf_report)

                re_metrics = compute_re_metrics(
                    y_true=y_true_arcs_types,
                    y_pred=y_pred_arcs_types,
                    no_rel_id=no_rel_id
                )

                eval_loss.append(np.mean(losses_tmp))
                eval_loss_ner.append(np.mean(losses_tmp_ner))
                eval_loss_re.append(np.mean(losses_tmp_re))

                eval_las.append(re_metrics.f1_arcs_types)
                eval_uas.append(re_metrics.f1_arcs)

                plot()
        plot()
        return clf_reports

    def _stacked_attention(self, x, config, mask):
        d_model = config["num_heads"] * config["head_dim"]
        x = tf.keras.layers.Dense(d_model)(x)
        for i in range(config["num_layers"]):
            attn = DotProductAttention(**config)
            x = attn(x, training=self.training_ph, mask=mask)
        return x

    def predict(self, examples, batch_size=128):
        y_pred = []
        for start in trange(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            y_pred += self._predict_batch(examples_batch)
        return y_pred

    @staticmethod
    def evaluate(y_true, y_pred):
        flags_las = []
        flags_uas = []
        for i, j in zip(y_true, y_pred):
            flags_las.append(i == j)
            flags_uas.append(i[:-1] == j[:-1])
        las = np.mean(flags_las)
        uas = np.mean(flags_uas)
        return las, uas

    def _predict_batch(self, examples):
        feed_dict = self._get_feed_dict(examples, training=False)
        # s_arc, s_type = self.sess.run([self.s_arc, self.s_type], feed_dict=feed_dict)
        preds = []
        # for i in range(len(examples)):
        #     length_i = examples[i].num_tokens
        #     s_arc_i = s_arc[i, :length_i, :length_i]
        #     s_type_i = s_type[i, :length_i, :length_i]
        #
        #     indices_dep = range(length_i)
        #     indices_head = mst(s_arc_i)
        #     indices_type = s_type_i[indices_dep, indices_head].argmax(-1)
        #
        #     preds_i = list(zip(indices_head, indices_type))
        #     preds.append(preds_i)
        return preds

    def _get_feed_dict(self, examples, training):
        # tokens
        pad = "[PAD]"
        tokens = [x.tokens for x in examples]
        sequence_len = [x.num_tokens for x in examples]
        num_tokens_max = max(sequence_len)
        tokens = [x + [pad] * (num_tokens_max - l) for x, l in zip(tokens, sequence_len)]

        # ner labels
        other_label_id = self.config["model"]["ner"]["other_label_id"]
        ner_labels = [x.labels + [other_label_id] * (num_tokens_max - l) for x, l in zip(examples, sequence_len)]

        # # entities
        # num_entities = [x.num_entities for x in examples]
        # num_entities_max = max(num_entities)
        # entity_starts = []
        # entity_ends = []
        # pad_start = pad_end = 0
        # for i, x in enumerate(examples):
        #     for start, end in x.entities:
        #         entity_starts.append((i, start))
        #         entity_ends.append((i, end))
        #     for _ in range(num_entities_max - x.num_entities):
        #         entity_starts.append((i, pad_start))
        #         entity_ends.append((i, pad_end))

        # arcs
        type_ids = [(i, *arc) for i, x in enumerate(examples) for arc in x.arcs]

        # feed_dict
        feed_dict = {
            self.tokens_ph: tokens,
            self.sequence_len_ph: sequence_len,
            self.ner_labels_ph: ner_labels,
            self.type_ids_ph: type_ids,
            self.training_ph: training
        }

        return feed_dict

    def _set_placeholders(self):
        self.tokens_ph = tf.placeholder(tf.string, shape=[None, None], name="tokens_ph")
        self.sequence_len_ph = tf.placeholder(tf.int32, shape=[None], name="sequence_len_ph")
        # self.num_entities_ph = tf.placeholder(tf.int32, shape=[None], name="num_entities_ph")
        self.ner_labels_ph = tf.placeholder(tf.int32, shape=[None, None], name="ner_ph")
        # self.entity_starts = tf.placeholder(tf.int32, shape=[None, 2], name="entity_starts_ph")  # [id_example, start]
        # self.entity_ends = tf.placeholder(tf.int32, shape=[None, 2], name="entity_ends_ph")  # [id_example, end]

        # # [N, num_entities, num_entities], (i, j, k) - номер отношения r(j, k) примера i
        # self.type_ids_ph = tf.placeholder(tf.int32, shape=[None, None, None], name="type_ids_ph")
        self.type_ids_ph = tf.placeholder(tf.int32, shape=[None, 4], name="type_ids_ph")

        self.training_ph = tf.placeholder(tf.bool, shape=None, name="training_ph")

    def _get_ner_loss(self, ner_logits):
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            inputs=ner_logits,
            tag_indices=self.ner_labels_ph,
            sequence_lengths=self.sequence_len_ph,
            transition_params=self.transition_params
        )
        loss = -tf.reduce_mean(log_likelihood)
        return loss

    def _get_re_loss(self, s_type):
        s_type_softmax = tf.nn.softmax(s_type, axis=-1)
        s_type_softmax_gather = tf.gather_nd(s_type_softmax, self.type_ids_ph)
        loss = -tf.reduce_mean(tf.log(s_type_softmax_gather))
        # ner loss значительно выше из-за crf ->
        # без данного множителя модель фактически будет учиться только решать ner
        loss *= self.config["model"]["re"]["loss_weight"]
        return loss

    def _set_train_op(self):
        if self.config["optimizer"]["accumulate_gradients"]:
            self._set_train_op_with_acc()
        else:
            self._set_train_op_wo_acc()

    def _set_train_op_wo_acc(self):
        tvars = tf.trainable_variables()
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(
            init_lr=self.config["optimizer"]["init_lr"],
            global_step=global_step,
            warmup_steps=self.config["optimizer"]["warmup_steps"]
        )
        optimizer = tf.train.AdamOptimizer(lr)
        grads = tf.gradients(self.loss_train, tvars)
        if self.config["optimizer"]["clip_grads"]:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config["optimizer"]["clip_norm"])
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    def _set_train_op_with_acc(self):
        tvars = tf.trainable_variables()
        accum_vars = [
            tf.get_variable(
                name=v.name.split(":")[0] + "/accum",
                shape=v.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()
            ) for v in tvars
        ]
        self.global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(init_lr=self.config["opt"]["init_lr"], global_step=self.global_step,
                         warmup_steps=self.config["opt"]["warmup_steps"])
        optimizer = tf.train.AdamOptimizer(lr)
        num_acc_steps = self.config["opt"]["num_accumulation_steps"] * 1.0
        grads = tf.gradients(self.loss_train / num_acc_steps, tvars)
        self.all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
        if self.config["opt"]["clip_grads"]:
            grads, _ = tf.clip_by_global_norm(
                grads,
                clip_norm=self.config["opt"]["clip_norm"],
                use_norm=tf.cond(
                    self.all_are_finite,
                    lambda: tf.global_norm(grads),
                    lambda: tf.constant(1.0)
                )
            )
        self.reset_op = [v.assign(tf.zeros_like(v)) for v in accum_vars]
        self.acc_op = [v.assign_add(g) for v, g in zip(accum_vars, grads)]
        self.train_op = optimizer.apply_gradients(zip(accum_vars, tvars), global_step=self.global_step)
        with tf.control_dependencies([self.train_op]):
            self.global_step.assign_add(1)

    def _get_entity_embeddings(self, x, label_ids, d_model):
        """
        x: tf.Tensor of shape [batch_size, num_tokens, d_model]
        """
        config_re_span_emb = self.config["model"]["re"]["span_embeddings"]

        start_ids = infer_entities_bounds(label_ids, bound_ids=tf.constant(config_re_span_emb["ner_entity_start_ids"]))
        end_ids = infer_entities_bounds(label_ids, bound_ids=tf.constant(config_re_span_emb["ner_entity_end_ids"]))

        batch_size = tf.shape(x)[0]
        emb_type = config_re_span_emb["type"]

        if emb_type == 0:

            x_span = tf.gather_nd(x, end_ids)  # [N * num_entities, D]
            x_span = tf.reshape(x_span, [batch_size, -1, d_model])  # [N, num_entities, D]

        elif emb_type == 1:
            one = tf.constant(1)
            x_i = tf.gather_nd(x, start_ids)  # [N * num_entities, D]
            x_i_minus_one = tf.gather_nd(x, start_ids - one)  # [N * num_entities, D]
            x_j = tf.gather_nd(x, end_ids)  # [N * num_entities, D]
            x_j_plus_one = tf.gather_nd(x, end_ids + one)  # [N * num_entities, D]

            d_model_half = d_model // 2
            x_start = x_j - x_i_minus_one
            x_start = x_start[..., :d_model_half]
            x_end = x_i - x_j_plus_one
            x_end = x_end[..., d_model_half:]

            x_span = tf.concat([x_start, x_end], axis=-1)  # [N * num_entities, D]
            x_span = tf.reshape(x_span, [batch_size, -1, d_model])  # [N, num_entities, D]
            self.x_span = x_span
        else:
            raise ValueError(f"expected span_emb type in {{0, 1}}, got {emb_type}")

        return x_span
