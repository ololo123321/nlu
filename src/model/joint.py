import tensorflow as tf

from src.model.base import BaseModel


class BaseModelJoint(BaseModel):
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

    def _build_graph(self):
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            # ner
            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                num_labels = self.config["model"]["ner"]["num_labels"]
                self.dense_ner_labels = tf.keras.layers.Dense(num_labels)

                self.ner_logits_train, _, self.transition_params = self._build_ner_head(bert_out=bert_out_train)
                _, self.ner_preds_inference, _ = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    bert_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_labels, bert_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=self.ner_labels_ph
                )
                re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=self.ner_labels_ph
                )
                re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=self.ner_preds_inference
                )

                self.re_labels_true_entities = tf.argmax(re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
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

        loss = 0.0
        loss_ner = 0.0
        loss_re = 0.0
        num_batches = 0

        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict = self._get_feed_dict(examples_batch, mode=ModeKeys.VALID)
            loss_i, loss_ner_i, loss_re_i, ner_labels_pred, rel_labels_pred, num_entities = self.sess.run(
                [
                    self.loss,
                    self.loss_ner,
                    self.loss_re,
                    self.ner_preds_inference,
                    self.re_labels_true_entities,
                    self.num_entities
                ],
                feed_dict=feed_dict
            )
            loss += loss_i
            loss_ner += loss_ner_i
            loss_re += loss_re_i

            for i, x in enumerate(examples_batch):
                # ner
                y_true_ner_i = []
                y_pred_ner_i = []
                for j, t in enumerate(x.tokens):
                    y_true_ner_i.append(t.labels[0])
                    y_pred_ner_i.append(self.inv_ner_enc[ner_labels_pred[i, j]])
                y_true_ner.append(y_true_ner_i)
                y_pred_ner.append(y_pred_ner_i)

                # re
                num_entities_i = num_entities[i]
                # этот assert может не выполняться в случае, когда редкие сущности игнорятся
                # assert num_entities_i == len(x.entities), f"[{x.id}] {num_entities_i} != {len(x.entities)}"
                arcs_true = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    arcs_true[arc.head_index, arc.dep_index] = arc.rel_id

                arcs_pred = rel_labels_pred[i, :num_entities_i, :num_entities_i]
                y_true_re += [self.inv_re_enc[j] for j in arcs_true.flatten()]
                y_pred_re += [self.inv_re_enc[j] for j in arcs_pred.flatten()]

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches
        loss_ner /= num_batches
        loss_re /= num_batches

        # ner
        joiner = self.config["model"]["ner"]["prefix_joiner"]
        ner_metrics_entity_level = classification_report_ner(y_true=y_true_ner, y_pred=y_pred_ner, joiner=joiner)
        y_true_ner_flat = list(chain(*y_true_ner))
        y_pred_ner_flat = list(chain(*y_pred_ner))
        ner_metrics_token_level = classification_report(
            y_true=y_true_ner_flat, y_pred=y_pred_ner_flat, trivial_label="O"
        )

        # re
        re_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, trivial_label="O")

        # total
        # сделано так, чтобы случайный скор на таске с нулевым loss_coef не вносил подгрешность в score.
        # невозможность равенства нулю коэффициентов при лоссах на обоих тасках рассмотрена в BaseModel.__init__
        if self.config["model"]["ner"]["loss_coef"] == 0.0:
            score = re_metrics["micro"]["f1"]
        elif self.config["model"]["re"]["loss_coef"] == 0.0:
            score = ner_metrics_entity_level["micro"]["f1"]
        else:
            score = ner_metrics_entity_level["micro"]["f1"] * 0.5 + re_metrics["micro"]["f1"] * 0.5

        performance_info = {
            "ner": {
                "loss": loss_ner,
                "metrics": {
                    "entity_level": ner_metrics_entity_level,
                    "token_level": ner_metrics_token_level
                }
            },
            "re": {
                "loss": loss_re,
                "metrics": re_metrics,
            },
            "loss": loss,
            "score": score
        }

        return performance_info

    # TODO: реалзиовать случай window > 1
    def predict(self, examples: List[Example], window: int = 1, batch_size: int = 16, **kwargs):
        """
        инференс. примеры не должны содержать разметку токенов и пар сущностей!
        сделано так для того, чтобы не было непредсказуемых результатов.

        ner - запись лейблов в Token.labels
        re - создание новых инстансов Arc и запись их в Example.arcs
        """
        assert window == 1, "logic with window > 1 is not implemented :("

        # проверка примеров
        chunks = []
        for x in examples:
            assert len(x.arcs) == 0, f"[{x.id}] arcs are already annotated"
            assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for t in x.tokens:
                assert len(t.labels) == 0, f"[{x.id}] tokens are already annotated"
            chunks += x.chunks

        id2example = {x.id: x for x in examples}

        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunks_batch = chunks[start:end]
            feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.TEST)
            ner_labels_pred, rel_labels_pred, num_entities = self.sess.run(
                [self.ner_preds_inference, self.re_labels_pred_entities, self.num_entities_pred],
                feed_dict=feed_dict
            )

            m = max(len(x.tokens) for x in chunks_batch)
            assert m == ner_labels_pred.shape[1], f'{m} != {ner_labels_pred.shape[1]}'

            for i, chunk in enumerate(chunks_batch):
                example = id2example[chunk.parent]
                ner_labels_i = []
                # ner
                for j, t in enumerate(chunk.tokens):
                    id_label = ner_labels_pred[i, j]
                    label = self.inv_ner_enc[id_label]
                    ner_labels_i.append(label)

                entities_chunk = []
                tag2spans = get_entity_spans(labels=ner_labels_i, joiner=self.config["model"]["ner"]["prefix_joiner"])
                for label, spans in tag2spans.items():
                    for span in spans:
                        start_abs = chunk.tokens[span.start].index_abs
                        end_abs = chunk.tokens[span.end].index_abs
                        tokens = example.tokens[start_abs:end_abs + 1]
                        t_first = tokens[0]
                        t_last = tokens[-1]
                        text = example.text[t_first.span_rel.start:t_last.span_rel.end]
                        id_entity = 'T' + str(len(example.entities))
                        entity = Entity(
                            id=id_entity,
                            label=label,
                            text=text,
                            tokens=tokens,
                        )
                        example.entities.append(entity)
                        entities_chunk.append(entity)

                # re
                entities_sorted = sorted(entities_chunk, key=lambda e: (e.tokens[0].index_rel, e.tokens[-1].index_rel))
                arcs_pred = rel_labels_pred[i, :num_entities[i], :num_entities[i]]
                for j, k in zip(*np.where(arcs_pred != no_rel_id)):
                    id_label = arcs_pred[j, k]
                    id_head = entities_sorted[j].id
                    id_dep = entities_sorted[k].id
                    assert isinstance(id_head, str)  # чтоб не подсвечивалось жёлтым ниже
                    assert isinstance(id_dep, str)
                    id_arc = "R" + str(len(example.arcs))
                    arc = Arc(id=id_arc, head=id_head, dep=id_dep, rel=self.inv_re_enc[id_label])
                    example.arcs.append(arc)

    def reset_weights(self):
        super().reset_weights()

        bert_dir = self.config["model"]["bert"]["dir"]
        bert_scope = self.config["model"]["bert"]["scope"]
        var_list = {
            self._actual_name_to_checkpoint_name(x.name): x for x in tf.trainable_variables()
            if x.name.startswith(f"{self.model_scope}/{bert_scope}")
        }
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
        saver.restore(self.sess, checkpoint_path)

    def set_train_op_head(self):
        """
        [опционально] операция для предобучения только новых слоёв
        TODO: по-хорошему нужно global_step обновлять до нуля, если хочется продолжать обучение с помощью train_op.
         иначе learning rate будет считаться не совсем ожидаемо
        """
        tvars = [
            x for x in tf.trainable_variables()
            if x.name.startswith(f"{self.model_scope}/{self.ner_scope}")
            or x.name.startswith(f"{self.model_scope}/{self.re_scope}")
        ]
        opt = tf.train.AdamOptimizer()
        grads = tf.gradients(self.loss, tvars)
        self.train_op_head = opt.apply_gradients(zip(grads, tvars))

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

            ner_labels_i = []
            ptr = 1

            # tokens
            for t in x.tokens:
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.pieces)
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                label = t.labels[0]
                if mode != ModeKeys.TEST:
                    id_label = self.ner_enc[label]
                    ner_labels_i.append(id_label)  # ner решается на уровне токенов!
                ptr += num_pieces_ij

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            # relations
            if mode != ModeKeys.TEST:
                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    id_rel = self.re_enc[arc.rel]
                    re_labels.append((i, arc.head_index, arc.dep_index, id_rel))

            # write
            num_pieces.append(len(input_ids_i))
            num_tokens.append(len(x.tokens))
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

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0, 0))

        training = mode == ModeKeys.TRAIN

        d = {
            # bert
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,

            # ner
            self.first_pieces_coords_ph: first_pieces_coords,
            self.num_pieces_ph: num_pieces,
            self.num_tokens_ph: num_tokens,

            # common
            self.training_ph: training
        }

        if mode != ModeKeys.TEST:
            d[self.ner_labels_ph] = ner_labels
            d[self.re_labels_ph] = re_labels

        return d

    def _set_placeholders(self):
        # bert inputs
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        # ner inputs
        # [id_example, id_piece]
        self.first_pieces_coords_ph = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name="first_pieces_coords")
        self.num_pieces_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_pieces")
        self.num_tokens_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_tokens")
        self.ner_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ner_labels")

        # re inputs
        # [id_example, id_head, id_dep, id_rel]
        self.re_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, 4], name="re_labels")

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    def _set_loss(self):
        self.loss_ner = self._get_ner_loss()
        self.loss_re = self._get_re_loss()
        self.loss = self.loss_ner + self.loss_re

    def _set_train_op(self):
        num_samples = self.config["training"]["num_train_samples"]
        batch_size = self.config["training"]["batch_size"]
        num_epochs = self.config["training"]["num_epochs"]
        num_train_steps = int(num_samples / batch_size) * num_epochs
        warmup_proportion = self.config["optimizer"]["warmup_proportion"]
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        init_lr = self.config["optimizer"]["init_lr"]
        self.train_op = create_optimizer(
            loss=self.loss,
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False
        )

    def _build_bert(self, training):
        bert_dir = self.config["model"]["bert"]["dir"]
        bert_scope = self.config["model"]["bert"]["scope"]
        reuse = not training
        with tf.variable_scope(bert_scope, reuse=reuse):
            bert_config = BertConfig.from_json_file(os.path.join(bert_dir, "bert_config.json"))
            bert_config.attention_probs_dropout_prob = self.config["model"]["bert"]["attention_probs_dropout_prob"]
            bert_config.hidden_dropout_prob = self.config["model"]["bert"]["hidden_dropout_prob"]
            model = BertModel(
                config=bert_config,
                is_training=training,
                input_ids=self.input_ids_ph,
                input_mask=self.input_mask_ph,
                token_type_ids=self.segment_ids_ph
            )
            x = model.get_sequence_output()
        return x

    def _build_ner_head(self,  bert_out):
        """
        bert_out -> dropout -> stacked birnn (optional) -> dense(num_labels) -> crf (optional)
        :param bert_out:
        :return:
        """
        use_crf = self.config["model"]["ner"]["use_crf"]
        num_labels = self.config["model"]["ner"]["num_labels"]

        # dropout
        if (self.birnn_ner is None) or (self.config["model"]["ner"]["rnn"]["dropout"] == 0.0):
            x = self.bert_dropout(bert_out, training=self.training_ph)
        else:
            x = bert_out

        # birnn
        if self.birnn_ner is not None:
            sequence_mask = tf.sequence_mask(self.num_pieces_ph)
            x = self.birnn_ner(x, training=self.training_ph, mask=sequence_mask)

        # pieces -> tokens
        # сделано так для того, чтобы в ElmoJointModel не нужно было переопределять данный метод
        if self.first_pieces_coords_ph is not None:
            x = tf.gather_nd(x, self.first_pieces_coords_ph)  # [N, num_tokens_tokens, bert_dim or cell_dim * 2]

        # label logits
        logits = self.dense_ner_labels(x)

        # label ids
        if use_crf:
            with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
                transition_params = tf.get_variable("transition_params", [num_labels, num_labels], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, transition_params, self.num_tokens_ph)
        else:
            pred_ids = tf.argmax(logits, axis=-1)
            transition_params = None

        return logits, pred_ids, transition_params

    def _build_re_head(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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

    def _get_ner_loss(self):
        use_crf = self.config["model"]["ner"]["use_crf"]
        if use_crf:
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=self.ner_logits_train,
                tag_indices=self.ner_labels_ph,
                sequence_lengths=self.num_tokens_ph,
                transition_params=self.transition_params
            )
            loss = -tf.reduce_mean(log_likelihood)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.ner_labels_ph, logits=self.ner_logits_train
            )
            loss = tf.reduce_mean(loss)

        loss *= self.config["model"]["ner"]["loss_coef"]
        return loss

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

    def _actual_name_to_checkpoint_name(self, name: str) -> str:
        bert_scope = self.config["model"]["bert"]["scope"]
        prefix = f"{self.model_scope}/{bert_scope}/"
        name = name[len(prefix):]
        name = name.replace(":0", "")
        return name

    def _get_ner_embeddings(self, ner_labels):
        x_emb = self.ner_emb(ner_labels)
        if self.ner_emb_layer_norm is not None:
            x_emb = self.ner_emb_layer_norm(x_emb)
        x_emb = self.ner_emb_dropout(x_emb, training=self.training_ph)
        return x_emb
