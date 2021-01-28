import random
import os
from collections import defaultdict

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics import classification_report

from .utils import compute_f1, infer_entities_bounds
from .layers import DotProductAttention, GraphEncoder, GraphEncoderInputs
from .optimization import noam_scheme
from .preprocessing import Arc


# model

class JointModelV1:
    """
    Функционал:
    * ищется ровно одно событие (NER + построение маркированных рёбер event -> entity)

    Допущения:
    * NER для старых именных сущностей (ORG, PER, LOC, MEDIA, SPORT) уже решён
    """
    scope = "clf_head"

    def __init__(self, sess, config):
        """
        config = {
            "model": {
                "embedder": {},
                "entities": {},
                "relations": {},
                "event": {},
            },
            "optimizer": {}
        }
        """
        self.sess = sess
        self.config = config

        # placeholders
        self.tokens_ph = None
        self.sequence_len_ph = None
        self.num_events_ph = None
        self.num_other_entities_ph = None
        self.ner_labels_entities_ph = None
        self.ner_labels_event_ph = None
        self.type_ids_ph = None
        self.training_ph = None
        self.lr_ph = None

        # некоторые нужные тензоры
        self.event_logits = None   # для расчёта loss
        self.event_entity_role_logits = None   # для расчёта loss
        self.ner_labels_pred = None
        self.event_labels_pred = None
        self.rel_labels_pred_train = None  # зависит от: ner_labels_entities_ph, ner_labels_event_ph
        self.rel_labels_pred_inference = None  # зависит от: ner_labels_entities_ph, ner_labels_pred
        self.num_entities_inference = None
        self.num_events_inference = None
        self.loss_ner = None
        self.loss_event_entity = None
        self.loss = None

        # ops
        self.train_op = None
        self.acc_op = None
        self.reset_op = None

        # for debug
        self.event_emb_shape = None
        self.entity_emb_shape = None

    def build(self):
        self._set_placeholders()

        x = self._build_embedder()
        self._build_event_head(x)

        self._set_loss()
        self._set_train_op()

    def train(
            self,
            train_examples,
            eval_examples,
            no_rel_id: int,
            num_epochs=1,
            batch_size=128,
            train_op_name="train_op",
            id2label_ner=None,
            id2label_roles=None,
            checkpoint_path=None,
            event_tag="Bankruptcy"
    ):
        def print_clf_report(y_true, y_pred, id2label):
            if id2label is not None:
                target_names = [id2label[label] for label in sorted(set(y_true) | set(y_pred))]
            else:
                target_names = None
            print(classification_report(y_true, y_pred, target_names=target_names))

        # TODO: отделить конфигурацию оптимизатора от конфигурации обучения
        num_acc_steps = self.config["optimizer"]["num_accumulation_steps"]
        global_batch_size = batch_size * num_acc_steps
        epoch_steps = len(train_examples) // global_batch_size + 1
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
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            saver = tf.train.Saver(var_list)

        for step in range(num_train_steps):
            examples_batch = random.sample(train_examples, batch_size)
            feed_dict, _ = self._get_feed_dict(examples_batch, training=True)
            if self.config["optimizer"]["reduce_lr_on_plateau"]:
                feed_dict[self.lr_ph] = lr
            # event_emb_shape, entity_emb_shape = self.sess.run([self.event_emb_shape, self.entity_emb_shape], feed_dict=feed_dict)
            # print("event_emb_shape:", event_emb_shape)
            # print("entity_emb_shape:", entity_emb_shape)
            _, loss = self.sess.run([train_op, self.loss], feed_dict=feed_dict)
            # print(f"loss: {loss}")

            # if step % plot_step == 0:
            #     plot()

            if step != 0 and step % epoch_steps == 0:
                print(f"epoch {epoch} finished. evaluation starts.")
                y_true_ner = []
                y_pred_ner = []

                y_true_arcs_types = []
                y_pred_arcs_types = []

                for start in range(0, len(eval_examples), batch_size):
                    end = start + batch_size
                    examples_batch = eval_examples[start:end]
                    feed_dict, id2index = self._get_feed_dict(examples_batch, training=False)
                    loss, ner_labels_pred, rel_labels_pred = self.sess.run(
                        [self.loss, self.ner_labels_pred, self.rel_labels_pred_train],
                        feed_dict=feed_dict
                    )

                    for i, x in enumerate(examples_batch):
                        # TODO: рассмотреть случаи num_events == 0, num_entities_wo_events == 0
                        num_events = x.num_events
                        num_entities_wo_events = x.num_entities_wo_events

                        # events
                        for label_true, label_pred in zip(x.labels_events[event_tag], ner_labels_pred[i]):
                            y_true_ner.append(label_true)
                            y_pred_ner.append(label_pred)

                        # roles
                        arcs_true = np.full((num_events, num_entities_wo_events), no_rel_id, dtype=np.int32)
                        for arc in x.arcs:
                            id_head = id2index[(x.id, arc.head)]
                            id_dep = id2index[(x.id, arc.dep)]
                            arcs_true[id_head, id_dep] = arc.rel
                        arcs_pred = rel_labels_pred[i, :num_events, :num_entities_wo_events]
                        y_true_arcs_types.append(arcs_true.flatten())
                        y_pred_arcs_types.append(arcs_pred.flatten())

                # events
                print("ner result:")
                print_clf_report(y_true_ner, y_pred_ner, id2label=id2label_ner)

                # roles
                y_true_arcs_types = np.concatenate(y_true_arcs_types)
                y_pred_arcs_types = np.concatenate(y_pred_arcs_types)
                print("roles result:")
                print_clf_report(y_true_arcs_types, y_pred_arcs_types, id2label=id2label_roles)
                re_metrics = compute_f1(labels=y_true_arcs_types, preds=y_pred_arcs_types)

                print('evaluation results:')
                print(re_metrics)

                # TODO: учитывать качество решения задачи ner. мб использовать лог-лосс умарный?
                score = re_metrics['f1']

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

    def predict(self, examples, batch_size=128):
        filename2id_arc = defaultdict(int)
        events_counter = defaultdict(int)
        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict, id2index = self._get_feed_dict(examples_batch, training=False)
            num_events, num_entities, rel_labels_pred = self.sess.run(
                [self.num_events_inference, self.num_entities_inference, self.rel_labels_pred_inference],
                feed_dict=feed_dict
            )
            d = {(id_example, i): id_entity for (id_example, id_entity), i in id2index.items()}
            assert len(d) == len(id2index)
            mask = rel_labels_pred != 0
            for i, j, k in zip(*np.where(mask)):
                if j < num_events[i] and k < num_entities[i]:
                    x = examples[i]
                    id_head = f"E{events_counter[x.filename]}"
                    id_dep = d[(x.id, k)]  # компании не ищем, поэтому айдишник знаем
                    id_arc = filename2id_arc[x.filename]
                    arc = Arc(id=id_arc, head=id_head, dep=id_dep, rel=rel_labels_pred[i, j, k])
                    x.arcs.append(arc)
                    filename2id_arc[x.filename] += 1
                    events_counter[x.filename] += 1

    def restore(self, model_dir):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
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

    def _build_embedder(self):
        # embedder
        # TODO: добавить возможность выбора bert
        if self.config["model"]["embedder"]["type"] == "elmo":
            elmo = hub.Module(self.config["model"]["embedder"]["dir"], trainable=False)
            input_dict = {
                "tokens": self.tokens_ph,
                "sequence_len": self.sequence_len_ph
            }
            x = elmo(input_dict, signature="tokens", as_dict=True)["elmo"]  # [N, T, elmo_dim]

            elmo_dropout = tf.keras.layers.Dropout(self.config["model"]["embedder"]["dropout"])
            x = elmo_dropout(x, training=self.training_ph)
        else:
            raise NotImplementedError
        return x

    def _build_encoder_layers(self, x, config):
        """обучаемые с нуля верхние слои:"""
        sequence_mask = tf.sequence_mask(self.sequence_len_ph)
        d_model = None
        if config["attention"]["enabled"]:
            x = self._stacked_attention(x, config=config["attention"],
                                        mask=sequence_mask)  # [N, T, d_model]
            d_model = config["attention"]["num_heads"] * config["attention"]["head_dim"]
        if config["rnn"]["enabled"]:
            x = self._stacked_rnn(x, config=config["rnn"], mask=sequence_mask)
            d_model = config["rnn"]["cell_dim"] * 2
        if d_model is None:
            d_model = self.config["embedder"]["dim"]
        return x, d_model

    def _build_event_head(self, x):
        """
        elmo -> bilstm -> entity_embeddings, event_embeddings -> bilinear
        training - параметр, от которого зависит то, какие лейблы событий использовать: предсказанные или истинные
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            config_re = self.config["model"]["re"]

            # эмбеддинги лейблов именных сущностей
            # TODO: лучше не юзать, чтоб не зависеть от NER-лейблов именных сущностей.
            #  лучше от той головы брать только айдишники начала сущностей
            if config_re["ner_embeddings"]["use"]:
                ner_emb = tf.keras.layers.Embedding(
                    input_dim=config_re["ner_embeddings"]["num_labels"],
                    output_dim=config_re["ner_embeddings"]["dim"]
                )(self.ner_labels_entities_ph)
                ner_dropout = tf.keras.layers.Dropout(config_re["ner_embeddings"]["dropout"])
                ner_emb = ner_dropout(ner_emb, training=self.training_ph)

                # merged
                if config_re["merged_embeddings"]["merge_mode"] == "concat":
                    x = tf.concat([x, ner_emb], axis=-1)
                elif config_re["merged_embeddings"]["merge_mode"] == "sum":
                    # TODO: вставить assert на равенство размерности эмбеддингов сущностей и разметности elmo
                    x += ner_emb
                else:
                    raise NotImplementedError

                x = tf.keras.layers.Dropout(config_re["merged_embeddings"]["dropout"])(x, training=self.training_ph)

                if config_re["merged_embeddings"]["layernorm"]:
                    x = tf.keras.layers.LayerNormalization()(x)

            # обучаемые с нуля верхние слои:
            x, d_model = self._build_encoder_layers(x, config=self.config["model"]["embedder"])

            def get_embeddings(emb_token, coords):
                batch_size = tf.shape(emb_token)[0]
                res = tf.gather_nd(emb_token, coords)  # [N * num_entities_max, d_model]
                res = tf.reshape(res, [batch_size, -1, d_model])  # [N, num_entities_max, d_model]
                return res

            # векторы именных сущностей
            bound_ids_entity = tf.constant(self.config["model"]["ner"]["start_ids"], dtype=tf.int32)
            entity_coords, num_entities = infer_entities_bounds(label_ids=self.ner_labels_entities_ph, bound_ids=bound_ids_entity)
            entity_emb = get_embeddings(x, coords=entity_coords)
            self.num_entities_inference = num_entities  # пока не нужно, но пусть будет для общности

            # поиск триггеров событий
            num_labels_event = self.config["model"]["event"]["num_labels"]  # включая "O"
            self.event_logits = tf.keras.layers.Dense(num_labels_event)(x)
            self.ner_labels_pred = tf.argmax(self.event_logits, axis=-1, output_type=tf.int32)

            # векторы событий
            bound_ids_event = tf.constant(self.config["model"]["event"]["start_ids"], dtype=tf.int32)

            # train
            event_coords, _ = infer_entities_bounds(label_ids=self.ner_labels_event_ph, bound_ids=bound_ids_event)
            event_emb_train = get_embeddings(x, coords=event_coords)

            # inference
            event_coords, num_events = infer_entities_bounds(label_ids=self.ner_labels_pred, bound_ids=bound_ids_event)
            event_emb_inference = get_embeddings(x, coords=event_coords)
            self.num_events_inference = num_events

            # получение логитов пар (событие, именная сущность)
            head_dim = dep_dim = self.config["model"]["re"]["bilinear"]["hidden_dim"]
            pairs_encoder = GraphEncoder(
                num_mlp_layers=self.config["model"]["re"]["mlp"]["num_layers"],
                head_dim=head_dim,
                dep_dim=dep_dim,
                output_dim=self.config["model"]["event"]["num_labels"],
                dropout=self.config["model"]["re"]["mlp"]["dropout"],
                activation=self.config["model"]["re"]["mlp"]["activation"]
            )

            # train
            inputs = GraphEncoderInputs(head=event_emb_train, dep=entity_emb)
            self.event_entity_role_logits = pairs_encoder(inputs, training=self.training_ph)  # [N, num_events, num_entities, num_roles]
            self.rel_labels_pred_train = tf.argmax(self.event_entity_role_logits, axis=-1, output_type=tf.int32)

            # inference
            inputs = GraphEncoderInputs(head=event_emb_inference, dep=entity_emb)
            event_entity_role_logits_inference = pairs_encoder(inputs, training=self.training_ph)  # [N, num_events, num_entities, num_roles]
            self.rel_labels_pred_inference = tf.argmax(event_entity_role_logits_inference, axis=-1, output_type=tf.int32)

            # debug
            self.event_emb_shape = tf.shape(event_emb_train)
            self.entity_emb_shape = tf.shape(entity_emb)

    def _get_feed_dict(self, examples, training):
        # tokens
        pad = "[PAD]"
        tokens = [x.tokens for x in examples]
        sequence_len = [x.num_tokens for x in examples]
        num_tokens_max = max(sequence_len)
        tokens = [x + [pad] * (num_tokens_max - l) for x, l in zip(tokens, sequence_len)]

        # ner labels (entities)
        other_label_id = self.config["model"]["ner"]["other_label_id"]
        ner_labels = [x.labels + [other_label_id] * (num_tokens_max - l) for x, l in zip(examples, sequence_len)]
        num_entities_other = [x.num_entities_wo_events for x in examples]

        id2index = {}
        for i, x in enumerate(examples):
            assert x.id is not None
            idx_entity = 0
            idx_event = 0
            for j, entity in enumerate(x.entities):
                assert entity.id is not None
                if entity.is_event_trigger:
                    id2index[(x.id, entity.id)] = idx_event
                    idx_event += 1
                else:
                    id2index[(x.id, entity.id)] = idx_entity
                    idx_entity += 1

        # arcs
        type_ids = []
        for i, x in enumerate(examples):
            for arc in x.arcs:
                id_head = id2index[(x.id, arc.head)]
                id_dep = id2index[(x.id, arc.dep)]
                type_ids.append((i, id_head, id_dep, arc.rel))

        # # если в батче нет ни одного отношения, то не получится посчитать лосс.
        # # решение - добавить одно отношение с лейблом NO_RELATION
        # if len(type_ids) == 0:
        #     for i, x in enumerate(examples):
        #         if x.num_entities > 0:
        #             type_ids.append((i, 0, 0, self.config['model']['re']['no_rel_id']))
        #             break

        # feed_dict
        feed_dict = {
            self.tokens_ph: tokens,
            self.sequence_len_ph: sequence_len,
            self.ner_labels_entities_ph: ner_labels,
            self.num_other_entities_ph: num_entities_other,
            self.type_ids_ph: type_ids,
            self.training_ph: training
        }

        # if training:
            # ner labels (event)
        # TODO: при инференсе не знаем истинных лейблов событий! сделано пока без условия на training, чтоб ничего не ломалось
        event_tag = self.config["model"]["event"]["tag"]
        other_label_id = self.config["model"]["event"]["other_label_id"]
        feed_dict[self.ner_labels_event_ph] = [
            x.labels_events[event_tag] + [other_label_id] * (num_tokens_max - l)
            for x, l in zip(examples, sequence_len)
        ]
        feed_dict[self.num_events_ph] = [x.num_events for x in examples]

        # assert min(feed_dict[self.num_events_ph]) >= 1
        # assert min(num_entities_other) >= 1
        # print("min num events:", min(feed_dict[self.num_events_ph]))
        # print("max num events:", max(feed_dict[self.num_events_ph]))
        #
        # print("min num entities:", min(num_entities_other))
        # print("max num entities:", max(num_entities_other))
        # print("num type_ids:", len(type_ids))
        # print("type_ids:", type_ids)

        return feed_dict, id2index

    def _set_placeholders(self):
        # для elmo
        self.tokens_ph = tf.placeholder(tf.string, shape=[None, None], name="tokens_ph")
        self.sequence_len_ph = tf.placeholder(tf.int32, shape=[None], name="sequence_len_ph")

        # для эмбеддингов сущнсотей
        self.ner_labels_entities_ph = tf.placeholder(tf.int32, shape=[None, None], name="ner_labels_entities_ph")
        self.ner_labels_event_ph = tf.placeholder(tf.int32, shape=[None, None], name="ner_labels_event_ph")

        # для маскирования пар (событие, сущность)
        self.num_events_ph = tf.placeholder(tf.int32, shape=[None], name="num_events_ph")
        self.num_other_entities_ph = tf.placeholder(tf.int32, shape=[None], name="num_other_entities_ph")

        # для обучения re; [id_example, id_head, id_dep, id_rel]
        self.type_ids_ph = tf.placeholder(tf.int32, shape=[None, 4], name="type_ids_ph")

        # для включения / выключения дропаутов
        self.training_ph = tf.placeholder(tf.bool, shape=None, name="training_ph")

        self.lr_ph = tf.placeholder(tf.float32, shape=None, name="lr_ph")

    def _set_loss(self):
        """
        event_entity_role_logits - tf.Tensor of shape [batch_size, num_events, num_entities, num_relations]
        """
        # поиск событий
        logits_shape = tf.shape(self.event_entity_role_logits)  # [4]
        no_rel_id = self.config['model']['re']['no_rel_id']
        labels = tf.broadcast_to(no_rel_id, logits_shape[:3])  # [batch_size, num_events, num_entities]
        labels = tf.tensor_scatter_nd_update(
            tensor=labels,
            indices=self.type_ids_ph[:, :-1],
            updates=self.type_ids_ph[:, -1],
        )  # [batch_size, num_events, num_entities]
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=self.event_entity_role_logits
        )  # [batch_size, num_events, num_entities]
        mask_events = tf.cast(tf.sequence_mask(self.num_events_ph), tf.float32)  # [batch_size, num_events]
        mask_entities = tf.cast(tf.sequence_mask(self.num_other_entities_ph), tf.float32)  # [batch_size, num_entities]
        masked_per_example_loss = per_example_loss * tf.expand_dims(mask_events, 2) * tf.expand_dims(mask_entities, 1)  # [batch_size, num_events, num_entities]
        total_loss = tf.reduce_sum(masked_per_example_loss)  # None
        num_pairs = tf.cast(tf.reduce_sum(self.num_events_ph * self.num_other_entities_ph), tf.float32)
        self.loss_event_entity = total_loss / num_pairs

        # поиск триггеров событий
        # TODO: рассмотреть случай crf
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.ner_labels_event_ph,
            logits=self.event_logits
        )  # [batch_size, maxlen]
        mask = tf.cast(tf.sequence_mask(self.sequence_len_ph), tf.float32)  # [batch_size, maxlen]
        masked_per_example_loss = per_example_loss * mask  # [batch_size, maxlen]
        self.loss_ner = tf.reduce_sum(masked_per_example_loss) / tf.reduce_sum(mask)

        # общий лосс
        self.loss = self.loss_event_entity + self.loss_ner

    def _set_train_op(self):
        tvars = tf.trainable_variables()
        global_step = tf.train.get_or_create_global_step()
        if self.config['optimizer']['reduce_lr_on_plateau']:
            lr = self.lr_ph
        else:
            if self.config['optimizer']['noam_scheme']:
                lr = noam_scheme(
                    init_lr=self.config["optimizer"]["init_lr"],
                    global_step=global_step,
                    warmup_steps=self.config["optimizer"]["warmup_steps"]
                )
            else:
                lr = self.config['optimizer']['init_lr']
        optimizer = tf.train.AdamOptimizer(lr)
        grads = tf.gradients(self.loss, tvars)
        if self.config["optimizer"]["clip_grads"]:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config["optimizer"]["clip_norm"])
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    def _stacked_attention(self, x, config, mask):
        d_model = config["num_heads"] * config["head_dim"]
        x = tf.keras.layers.Dense(d_model)(x)
        for i in range(config["num_layers"]):
            attn = DotProductAttention(**config)
            x = attn(x, training=self.training_ph, mask=mask)
        return x

    def _stacked_rnn(self, x, config, mask):
        for i in range(config['num_layers']):
            with tf.variable_scope(f"recurrent_layer_{i}"):
                xi = self._bidirectional(x=x, config=config, mask=mask)
                if config['skip_connections']:
                    if i == 0:
                        x = xi
                    else:
                        x += xi
                else:
                    x = xi
        return x

    def _bidirectional(self, x, config, mask):
        cell_name = config["cell_name"]
        cell_dim = config["cell_dim"]
        dropout = config["dropout"]
        recurrent_dropout = config["recurrent_dropout"]

        if cell_name == "lstm":
            recurrent_layer = tf.keras.layers.LSTM
        elif cell_name == "gru":
            recurrent_layer = tf.keras.layers.GRU
        else:
            raise Exception(f"expected cell_name in {{lstm, gru}}, got {cell_name}")

        recurrent_layer = recurrent_layer(
            units=cell_dim,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            name=cell_name
        )
        bidirectional_layer = tf.keras.layers.Bidirectional(recurrent_layer, name="bidirectional")
        x = bidirectional_layer(x, mask=mask, training=self.training_ph)
        return x
