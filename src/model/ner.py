import os
import json
from typing import Dict, List
from abc import abstractmethod
from itertools import chain

from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer
import tensorflow as tf
import numpy as np

from src.data.base import Example, Entity
from src.data.postprocessing import get_valid_spans
from src.model.base import BaseModel, ModeKeys
from src.model.layers import StackedBiRNN, GraphEncoder, GraphEncoderInputs
from src.model.utils import get_dense_labels_from_indices, upper_triangular
from src.metrics import classification_report, classification_report_ner
from src.utils import get_entity_spans, batches_gen


class BaseModelNER(BaseModel):
    ner_scope = "ner"

    def __init__(self, sess, config: Dict = None, ner_enc: Dict = None):
        super().__init__(sess=sess, config=config)
        self._ner_enc = None
        self._inv_ner_enc = None

        self.ner_enc = ner_enc

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.ner_scope):
            self._build_ner_head()

    def save_encoders(self, model_dir: str):
        with open(os.path.join(model_dir, "ner_enc.json"), "w") as f:
            json.dump(self.ner_enc, f, indent=4)

    def load_encoders(self, model_dir: str):
        with open(os.path.join(model_dir, "ner_enc.json")) as f:
            self.ner_enc = json.load(f)

    @abstractmethod
    def _build_ner_head(self):
        pass

    @property
    def ner_enc(self):
        return self._ner_enc

    @property
    def inv_ner_enc(self):
        return self._inv_ner_enc

    @ner_enc.setter
    def ner_enc(self, ner_enc: Dict):
        self._ner_enc = ner_enc
        if ner_enc is not None:
            self._inv_ner_enc = {v: k for k, v in ner_enc.items()}


class BertBaseModelNER(BaseModelNER):
    def __init__(self, sess, config: dict = None, ner_enc: Dict = None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc)

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

        self.bert_out_train = None
        self.bert_out_pred = None

    def _build_embedder(self):
        self.bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
        self.bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

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

    def _actual_name_to_checkpoint_name(self, name: str) -> str:
        bert_scope = self.config["model"]["bert"]["scope"]
        prefix = f"{self.model_scope}/{bert_scope}/"
        name = name[len(prefix):]
        name = name.replace(":0", "")
        return name

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

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    def reset_weights(self, scope: str = None):
        super().reset_weights(scope=scope)

        bert_dir = self.config["model"]["bert"]["dir"]
        bert_scope = self.config["model"]["bert"]["scope"]
        var_list = {
            self._actual_name_to_checkpoint_name(x.name): x for x in tf.trainable_variables()
            if x.name.startswith(f"{self.model_scope}/{bert_scope}")
        }
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
        saver.restore(self.sess, checkpoint_path)

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


class BertForFlatNER(BertBaseModelNER):
    """
    bert -> [bilstm x N] -> logits -> [crf]
    """

    def __init__(self, sess=None, config=None, ner_enc=None):
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
            },
            "training": {
                "num_epochs": 100,
                "batch_size": 16,
                "max_epochs_wo_improvement": 10
            },
            "inference": {
                "window": 1,
                "max_tokens_per_batch: 10000
            },
            "optimizer": {
                "init_lr": 2e-5,
                "num_train_steps": 100000,
                "num_warmup_steps": 10000
            }
        }
        """
        super().__init__(sess=sess, config=config, ner_enc=ner_enc)

        # TENSORS
        self.ner_logits_train = None
        self.transition_params = None
        self.ner_preds_inference = None

        # LAYERS
        self.bert_dropout = None
        self.birnn_ner = None
        self.dense_ner_labels = None

    def _build_ner_head(self):
        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

        if self.config["model"]["ner"]["use_birnn"]:
            self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

        self.dense_ner_labels = tf.keras.layers.Dense(self.config["model"]["ner"]["num_labels"])

        self.ner_logits_train, _, self.transition_params = self._build_ner_head_fn(bert_out=self.bert_out_train)
        _, self.ner_preds_inference, _ = self._build_ner_head_fn(bert_out=self.bert_out_pred)

    # TODO: профилирование!!!
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        chunks = []
        for x in examples:
            assert len(x.chunks) > 0
            chunks += x.chunks

        y_true_ner = []
        y_pred_ner = []

        loss = 0.0
        num_batches = 0

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            loss_i, ner_labels_pred = self.sess.run([self.loss, self.ner_preds_inference], feed_dict=feed_dict)
            loss += loss_i

            for i, x in enumerate(batch):
                y_true_ner_i = []
                y_pred_ner_i = []
                for j, t in enumerate(x.tokens):
                    y_true_ner_i.append(t.labels[0])
                    y_pred_ner_i.append(self.inv_ner_enc[ner_labels_pred[i, j]])
                y_true_ner.append(y_true_ner_i)
                y_pred_ner.append(y_pred_ner_i)

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches

        # ner
        joiner = self.config["model"]["ner"]["prefix_joiner"]
        ner_metrics_entity_level = classification_report_ner(y_true=y_true_ner, y_pred=y_pred_ner, joiner=joiner)
        y_true_ner_flat = list(chain(*y_true_ner))
        y_pred_ner_flat = list(chain(*y_pred_ner))
        ner_metrics_token_level = classification_report(
            y_true=y_true_ner_flat, y_pred=y_pred_ner_flat, trivial_label="O"
        )

        score = ner_metrics_entity_level["micro"]["f1"]
        performance_info = {
            "loss": loss,
            "score": score,
            "metrics": {
                "entity_level": ner_metrics_entity_level,
                "token_level": ner_metrics_token_level
            }
        }

        return performance_info

    # TODO: реалзиовать случай window > 1
    def predict(self, examples: List[Example], **kwargs) -> None:
        """
        инференс. примеры не должны содержать разметку токенов и пар сущностей!
        сделано так для того, чтобы не было непредсказуемых результатов.

        ner - запись лейблов в Token.labels
        re - создание новых инстансов Arc и запись их в Example.arcs
        """
        # проверка примеров
        chunks = []
        for x in examples:
            assert len(x.arcs) == 0, f"[{x.id}] arcs are already annotated"
            assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for t in x.tokens:
                assert len(t.labels) == 0, f"[{x.id}] tokens are already annotated"
            chunks += x.chunks

        id2example = {x.id: x for x in examples}

        gen = batches_gen(examples=chunks, max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"])
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.TEST)
            ner_labels_pred = self.sess.run(self.ner_preds_inference, feed_dict=feed_dict)

            m = max(len(x.tokens) for x in batch)
            assert m == ner_labels_pred.shape[1], f'{m} != {ner_labels_pred.shape[1]}'

            for i, chunk in enumerate(batch):
                example = id2example[chunk.parent]
                ner_labels_i = []
                for j, t in enumerate(chunk.tokens):
                    id_label = ner_labels_pred[i, j]
                    label = self.inv_ner_enc[id_label]
                    ner_labels_i.append(label)

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

    def _get_feed_dict(self, examples: List[Example], mode: str):
        assert self.ner_enc is not None

        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

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

        return d

    def _set_placeholders(self):
        super()._set_placeholders()
        self.ner_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ner_labels")

    def _set_loss(self):
        use_crf = self.config["model"]["ner"]["use_crf"]
        if use_crf:
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=self.ner_logits_train,
                tag_indices=self.ner_labels_ph,
                sequence_lengths=self.num_tokens_ph,
                transition_params=self.transition_params
            )
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.ner_labels_ph, logits=self.ner_logits_train
            )
            self.loss = tf.reduce_mean(loss)

    def _build_ner_head_fn(self,  bert_out):
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


class BertForNestedNER(BertBaseModelNER):
    def __init__(self, sess=None, config=None, ner_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc)

        self.tokens_pair_enc = None
        self.ner_logits_inference = None

    def _build_ner_head(self):
        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

        if self.config["model"]["ner"]["use_birnn"]:
            self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

        self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

        self.ner_logits_train = self._build_ner_head_fn(bert_out=self.bert_out_train)
        self.ner_logits_inference = self._build_ner_head_fn(bert_out=self.bert_out_pred)

    def _set_placeholders(self):
        super()._set_placeholders()
        self.ner_labels_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, 4], name="ner_labels"
        )  # [id_example, start, end, label]

    def _build_ner_head_fn(self,  bert_out):
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        if self.birnn_ner is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_ner(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]

        # encoding of pairs
        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.tokens_pair_enc(inputs=inputs, training=self.training_ph)  # [N, num_tok, num_tok, num_entities]
        return logits

    def _set_loss(self, *args, **kwargs):
        """"
        1 1 1
        0 1 1
        0 0 1
        i - start, j - end
        """
        # per example loss
        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        logits_shape = tf.shape(self.ner_logits_train)
        labels_shape = logits_shape[:3]
        labels = get_dense_labels_from_indices(indices=self.ner_labels_ph, shape=labels_shape, no_label_id=no_entity_id)
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.ner_logits_train
        )  # [batch_size, num_tokens, num_tokens]

        # mask
        maxlen = logits_shape[1]
        span_mask = upper_triangular(maxlen, dtype=tf.float32)
        sequence_mask = tf.sequence_mask(self.num_tokens_ph, dtype=tf.float32)  # [batch_size, num_tokens]
        mask = span_mask[None, :, :] * sequence_mask[:, None, :] * sequence_mask[:, :, None]  # [batch_size, num_tokens, num_tokens]

        masked_per_example_loss = per_example_loss * mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_valid_spans = tf.cast(tf.reduce_sum(mask), tf.float32)
        self.loss = total_loss / num_valid_spans

    def _get_feed_dict(self, examples: List[Example], mode: str):
        assert self.ner_enc is not None

        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

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

            # ner
            for entity in x.entities:
                start = entity.tokens[0].index_rel
                end = entity.tokens[-1].index_rel
                id_label = self.ner_enc[entity.label]
                ner_labels.append((i, start, end, id_label))

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

        return d

    # TODO: профилирование!!!
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        chunks = []
        for x in examples:
            assert len(x.chunks) > 0
            chunks += x.chunks

        y_true_ner = []
        y_pred_ner = []

        loss = 0.0
        num_batches = 0
        no_entity_id = 0  # TODO: брать из конфига

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            loss_i, ner_logits = self.sess.run([self.loss, self.ner_logits_inference], feed_dict=feed_dict)
            loss += loss_i

            for i, x in enumerate(batch):
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

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches

        # ner
        ner_metrics_entity_level = classification_report(y_true=y_true_ner, y_pred=y_pred_ner, trivial_label=no_entity_id)

        score = ner_metrics_entity_level["micro"]["f1"]
        performance_info = {
            "loss": loss,
            "score": score,
            "metrics": {
                "entity_level": ner_metrics_entity_level
            }
        }

        return performance_info

    # TODO: реалзиовать!
    def predict(self, examples: List[Example], **kwargs) -> None:
        pass


if __name__ == "__main__":
    model = BertForFlatNER()
    model = BertForNestedNER()
