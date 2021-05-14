from typing import Dict, List

import tensorflow as tf

from src.data.base import Example, Entity
from src.data.postprocessing import get_valid_spans
from src.model.base import BaseModelNER, BaseModelBert, ModeKeys
from src.model.layers import GraphEncoder, GraphEncoderInputs
from src.model.utils import upper_triangular
from src.metrics import classification_report, classification_report_ner
from src.utils import get_entity_spans, batches_gen, get_filtered_by_length_chunks, log


class BertForNerAsSequenceLabeling(BaseModelNER, BaseModelBert):
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
        self.total_loss = None
        self.loss_denominator = None

        # LAYERS
        self.dense_ner_labels = None

    def _build_ner_head(self):
        self.ner_logits_train, _, self.transition_params = self._build_ner_head_fn(bert_out=self.bert_out_train)
        _, self.ner_preds_inference, _ = self._build_ner_head_fn(bert_out=self.bert_out_pred)

    def _set_layers(self):
        super()._set_layers()
        self.dense_ner_labels = tf.keras.layers.Dense(self.config["model"]["ner"]["num_labels"])

    @log
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        maxlen = self.config["inference"]["maxlen"]
        chunks = get_filtered_by_length_chunks(examples=examples, maxlen=maxlen, pieces_level=self._is_bpe_level)

        y_true = []
        y_true_flat = []
        y_pred = []
        y_pred_flat = []
        total_loss = 0.0
        loss_denominator = 0

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=self._is_bpe_level
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            total_loss_i, d, ner_labels_pred = self.sess.run(
                [self.total_loss, self.loss_denominator, self.ner_preds_inference], feed_dict=feed_dict
            )
            total_loss += total_loss_i
            loss_denominator += d

            for i, x in enumerate(batch):
                y_true_i = []
                y_pred_i = []
                for j, t in enumerate(x.tokens):
                    y_true_i.append(t.label)
                    y_pred_i.append(self.inv_ner_enc[ner_labels_pred[i, j]])
                y_true.append(y_true_i)
                y_true_flat += y_true_i
                y_pred.append(y_pred_i)
                y_pred_flat += y_pred_i

        # loss
        loss = total_loss / loss_denominator

        # ner
        ner_metrics_entity_level = classification_report_ner(
            y_true=y_true, y_pred=y_pred, joiner=self.config["model"]["ner"]["prefix_joiner"]
        )
        ner_metrics_token_level = classification_report(
            y_true=y_true_flat, y_pred=y_pred_flat, trivial_label="O"
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
    # TODO: bug: при текущей логике обработки токенов, которые не удаётся разбить на кусочки
    #  теряется отображение "один к одному" между chunk.tokens и ner_labels_i,
    #  что влечёт поехавшие предикты. аналогичная проблема в evaluate
    @log
    def predict(self, examples: List[Example], **kwargs) -> None:
        """
        инференс. примеры не должны содержать разметку токенов и пар сущностей!
        сделано так для того, чтобы не было непредсказуемых результатов.

        ner - запись лейблов в Token.labels
        re - создание новых инстансов Arc и запись их в Example.arcs
        """
        maxlen = self.config["inference"]["maxlen"]
        chunks = get_filtered_by_length_chunks(examples=examples, maxlen=maxlen, pieces_level=self._is_bpe_level)

        # проверка примеров
        for x in chunks:
            assert x.parent is not None, f"[{x.id}] parent is not set. " \
                f"It is not a problem, but must be set for clarity"
            for t in x.tokens:
                assert t.label is None, f"[{x.id}] tokens are already annotated"

        id2example = {x.id: x for x in examples}
        assert len(id2example) == len(examples), f"examples must have unique ids, " \
            f"but got {len(id2example)} unique ids among {len(examples)} examples"

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=self._is_bpe_level
        )
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
        assert len(examples) > 0
        assert self.ner_enc is not None

        bert_inputs = self._get_bert_input_for_feed_dict(examples)

        d = {
            self.input_ids_ph: bert_inputs.input_ids,
            self.input_mask_ph: bert_inputs.input_mask,
            self.segment_ids_ph: bert_inputs.segment_ids,
            self.first_pieces_coords_ph: bert_inputs.first_pieces_coords,
            self.num_pieces_ph: bert_inputs.num_pieces,
            self.num_tokens_ph: bert_inputs.num_tokens,
            self.training_ph: mode == ModeKeys.TRAIN
        }

        if mode != ModeKeys.TEST:
            ner_labels = []
            for x in examples:
                ner_labels_i = []
                for t in x.tokens:
                    id_label = self.ner_enc[t.label]
                    ner_labels_i.append(id_label)  # ner решается на уровне токенов!
                ner_labels.append(ner_labels_i)

            num_tokens_max = max(bert_inputs.num_tokens)
            pad_label_id = self.config["model"]["ner"]["no_entity_id"]
            for i in range(len(examples)):
                ner_labels[i] += [pad_label_id] * (num_tokens_max - bert_inputs.num_tokens[i])

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
            per_example_loss = -log_likelihood
            total_loss = tf.reduce_sum(per_example_loss)
            num_sequences = tf.shape(self.ner_logits_train)[0]
            self.loss = total_loss / tf.cast(num_sequences, tf.float32)
            self.total_loss = total_loss
            self.loss_denominator = num_sequences
        else:
            per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.ner_labels_ph, logits=self.ner_logits_train
            )
            mask = tf.sequence_mask(self.num_tokens_ph, dtype=tf.float32)
            masked_per_example_loss = per_example_loss * mask
            total_loss = tf.reduce_sum(masked_per_example_loss)
            total_num_tokens = tf.reduce_sum(self.num_tokens_ph)
            self.loss = total_loss / tf.cast(total_num_tokens, tf.float32)
            self.total_loss = total_loss
            self.loss_denominator = total_num_tokens

    def _build_ner_head_fn(self,  bert_out):
        """
        bert_out -> dropout -> stacked birnn (optional) -> dense(num_labels) -> crf (optional)
        :param bert_out: [batch_size, num_pieces, D]
        :return:
        """
        use_crf = self.config["model"]["ner"]["use_crf"]
        num_labels = self.config["model"]["ner"]["num_labels"]

        x = self._get_token_level_embeddings(bert_out=bert_out)  # [batch_size, num_tokens, bert_dim or cell_dim * 2]

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


class BertForNerAsDependencyParsing(BaseModelNER, BaseModelBert):
    """
    https://arxiv.org/abs/2005.07150
    """
    def __init__(self, sess=None, config=None, ner_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc)

        # TENSORS
        self.tokens_pair_enc = None
        self.ner_logits_inference = None
        self.total_loss = None
        self.loss_denominator = None

        # LAYERS
        self.bert_dropout = None

    def _build_ner_head(self):
        self.ner_logits_train = self._build_ner_head_fn(bert_out=self.bert_out_train)
        self.ner_logits_inference = self._build_ner_head_fn(bert_out=self.bert_out_pred)

    def _set_placeholders(self):
        super()._set_placeholders()
        # [id_example, start, end, label]
        self.ner_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, 4], name="ner_labels")

    def _set_layers(self):
        super()._set_layers()
        self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

    def _build_ner_head_fn(self,  bert_out):
        x = self._get_token_level_embeddings(bert_out=bert_out)
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
        # no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        assert self.config["model"]["ner"]["no_entity_id"] == 0
        logits_shape = tf.shape(self.ner_logits_train)
        # labels_shape = logits_shape[:3]
        # labels = get_dense_labels_from_indices(indices=self.ner_labels_ph, shape=labels_shape, no_label_id=no_entity_id)
        # сделано так, чтобы уйти от использования функции get_dense_labels_from_indices
        labels = tf.scatter_nd(
            indices=self.ner_labels_ph[:, :-1], updates=self.ner_labels_ph[:, -1], shape=logits_shape[:-1]
        )  # [batch_size, num_tokens, num_tokens]
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
        self.total_loss = total_loss
        self.loss_denominator = num_valid_spans

    def _get_feed_dict(self, examples: List[Example], mode: str):
        assert len(examples) > 0
        assert self.ner_enc is not None

        bert_inputs = self._get_bert_input_for_feed_dict(examples)

        d = {
            # bert
            self.input_ids_ph: bert_inputs.input_ids,
            self.input_mask_ph: bert_inputs.input_mask,
            self.segment_ids_ph: bert_inputs.segment_ids,

            # ner
            self.first_pieces_coords_ph: bert_inputs.first_pieces_coords,
            self.num_pieces_ph: bert_inputs.num_pieces,
            self.num_tokens_ph: bert_inputs.num_tokens,

            # common
            self.training_ph: mode == ModeKeys.TRAIN
        }

        if mode != ModeKeys.TEST:
            ner_labels = []
            for i, x in enumerate(examples):
                for entity in x.entities:
                    assert entity.label is not None
                    start = entity.tokens[0].index_rel
                    assert start is not None
                    end = entity.tokens[-1].index_rel
                    assert end is not None
                    id_label = self.ner_enc[entity.label]
                    ner_labels.append((i, start, end, id_label))

            if len(ner_labels) == 0:
                ner_labels.append((0, 0, 0, 0))

            d[self.ner_labels_ph] = ner_labels

        return d

    @log
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        chunks = []
        for x in examples:
            assert len(x.chunks) > 0
            chunks += x.chunks

        y_true = []
        y_pred = []

        total_loss = 0.0
        loss_denominator = 0
        no_entity_label = "O"  # TODO: брать из конфига

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            total_loss_i, d, ner_logits = self.sess.run([self.total_loss, self.loss_denominator, self.ner_logits_inference], feed_dict=feed_dict)
            total_loss += total_loss_i
            loss_denominator += d

            for i, x in enumerate(batch):
                num_tokens = len(x.tokens)
                num_tokens_squared = num_tokens ** 2

                y_true_i = [no_entity_label] * num_tokens_squared
                for entity in x.entities:
                    start = entity.tokens[0].index_rel
                    end = entity.tokens[-1].index_rel
                    y_true_i[num_tokens * start + end] = entity.label
                y_true += y_true_i

                y_pred_i = [no_entity_label] * num_tokens_squared
                ner_logits_i = ner_logits[i, :num_tokens, :num_tokens, :]
                spans_filtered = get_valid_spans(logits=ner_logits_i,  is_flat_ner=False)
                for span in spans_filtered:
                    y_pred_i[num_tokens * span.start + span.end] = self.inv_ner_enc[span.label]
                y_pred += y_pred_i

        loss = total_loss / loss_denominator
        ner_metrics_entity_level = classification_report(y_true=y_true, y_pred=y_pred, trivial_label=no_entity_label)
        score = ner_metrics_entity_level["micro"]["f1"]
        performance_info = {
            "loss": loss,
            "score": score,
            "metrics": ner_metrics_entity_level
        }

        return performance_info

    # TODO: реалзиовать случай window > 1
    # TODO: копипаста в начале с BertForFlatNER
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
            # assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for chunk in x.chunks:
                assert chunk.parent is not None, f"[{x.id}] parent for chunk {chunk.id} is not set. " \
                    f"It is not a problem, but must be set for clarity"
                chunks.append(chunk)

        id2example = {x.id: x for x in examples}
        assert len(id2example) == len(examples), f"examples must have unique ids, " \
            f"but got {len(id2example)} unique ids among {len(examples)} examples"

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.TEST)
            ner_logits = self.sess.run(self.ner_logits_inference, feed_dict=feed_dict)

            for i, chunk in enumerate(batch):
                example = id2example[chunk.parent]
                num_tokens_i = len(chunk.tokens)

                ner_logits_i = ner_logits[i, :num_tokens_i, :num_tokens_i, :]
                spans_filtered = get_valid_spans(logits=ner_logits_i, is_flat_ner=False)
                for span in spans_filtered:
                    start_abs = chunk.tokens[span.start].index_abs
                    end_abs = chunk.tokens[span.end].index_abs
                    tokens = example.tokens[start_abs:end_abs + 1]
                    t_first = tokens[0]
                    t_last = tokens[-1]
                    text = example.text[t_first.span_abs.start:t_last.span_abs.end]
                    id_entity = 'T' + str(len(example.entities))
                    entity = Entity(
                        id=id_entity,
                        label=span.label,
                        text=text,
                        tokens=tokens,
                    )
                    example.entities.append(entity)
