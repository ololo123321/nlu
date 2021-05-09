from typing import Dict, List
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
    def __init__(self, sess: tf.Session = None, config: Dict = None, re_enc: Dict = None):
        super().__init__(sess=sess, config=config, re_enc=re_enc)

        # TENSORS
        self.ner_logits_train = None
        self.transition_params = None
        self.ner_preds_inference = None
        self.total_loss = None
        self.loss_denominator = None

        # LAYERS
        self.dense_ner_labels = None

    def _build_re_head(self):
        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

        if self.config["model"]["ner"]["use_birnn"]:
            self.birnn_bert = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

        self.dense_ner_labels = tf.keras.layers.Dense(self.config["model"]["ner"]["num_labels"])

        self.ner_logits_train, _, self.transition_params = self._build_ner_head_fn(bert_out=self.bert_out_train)
        _, self.ner_preds_inference, _ = self._build_ner_head_fn(bert_out=self.bert_out_pred)

    def _set_layers(self):
        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

        if self.config["model"]["ner"]["use_birnn"]:
            self.birnn_bert = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

    # TODO: профилирование!!!
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        maxlen = self.config["inference"]["maxlen"]
        chunks = get_filtered_by_length_chunks(examples=examples, maxlen=maxlen, pieces_level=self._is_bpe_level)

        y_true_ner = []
        y_pred_ner = []
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
                y_true_ner_i = []
                y_pred_ner_i = []
                for j, t in enumerate(x.tokens):
                    y_true_ner_i.append(t.labels[0])
                    y_pred_ner_i.append(self.inv_ner_enc[ner_labels_pred[i, j]])
                y_true_ner.append(y_true_ner_i)
                y_pred_ner.append(y_pred_ner_i)

        # loss
        loss = total_loss / loss_denominator

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
        maxlen = self.config["inference"]["maxlen"]
        chunks = get_filtered_by_length_chunks(examples=examples, maxlen=maxlen, pieces_level=self._is_bpe_level)

        # проверка примеров
        for x in chunks:
            assert x.parent is not None, f"[{x.id}] parent is not set. " \
                f"It is not a problem, but must be set for clarity"
            for t in x.tokens:
                assert len(t.labels) == 0, f"[{x.id}] tokens are already annotated"

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
                assert len(t.token_ids) > 0
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.token_ids)
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                if mode != ModeKeys.TEST:
                    label = t.labels[0]
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
