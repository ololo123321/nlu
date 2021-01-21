import numpy as np
import tensorflow as tf
from collections import namedtuple
from sklearn.metrics import f1_score


# REScores = namedtuple("REScores", ["f1_arcs", "f1_arcs_types"])
# NERScores = namedtuple("NERScores", ["f1_labels", "f1_entities"])


def compute_f1(preds, labels):
    """
    https://github.com/facebookresearch/SpanBERT/blob/10641ea3795771dd96e9e3e9ef0ead4f4f6a29d2/code/run_tacred.py#L245
    :param preds:
    :param labels:
    :return:
    """
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


# def compute_re_metrics(y_true: np.ndarray, y_pred: np.ndarray, no_rel_id: int) -> REScores:
#     """
#     Используются следующие метрики для оценки качества решения задачи relation extraction:
#     * f1 бинарного классификатора, определяющего наличие произвольного семанического отншения r(i, j) между
#       именными сущносями i и j.
#     * f1-micro многоклассового классификтора, определяющего наличие семантического отношения r(i, j) между
#       именныыми сущносями i и j.
#
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1 = 2 * precision * recall / (precision + recall)
#     """
#     mask_arcs_true = y_true != no_rel_id
#     mask_arcs_pred = y_pred != no_rel_id
#
#     # f1 {связаны ли семантически сущности i, j}
#     precision = mask_arcs_true[mask_arcs_pred].mean()
#     recall = mask_arcs_pred[mask_arcs_true].mean()
#     f1_arcs = f1(precision, recall)
#
#     # f1 {связаны ли семантически сущности i, j отношением r}
#
#     # true_pos_sum = mask_arcs_true.sum()  # сколько всего непустых отношений
#     # pred_pos_sum = mask_arcs_pred.sum()  # сколько раз предсказали непустое отношени
#     # # как много раз верно предсказали истинное непустое отношение:
#     # tp_sum = ((y_true == y_pred) & mask_arcs_true).sum()
#     # # расчёт f1:
#     # precision = tp_sum / pred_pos_sum
#     # recall = tp_sum / true_pos_sum
#     # f1_arcs_types = f1(precision, recall)
#
#     f1_arcs_types = compute_f1_micro_wo_trivial(y_true=y_true, y_pred=y_pred, no_rel_id=no_rel_id)
#     return REScores(f1_arcs=f1_arcs, f1_arcs_types=f1_arcs_types)
#
#
# def compute_ner_metrics(y_true: np.ndarray, y_pred: np.ndarray, other_label_id: int):
#     """
#     Используются следующие метрики для оценки качества решения задачи NER:
#     * f1 многоклассового классификатора, сопоставляющего каждому токену лейбл именной сущности
#     * поспановая f1 качества предсказания именных сущностей
#     """
#     pass


def infer_entities_bounds(label_ids: tf.Tensor, bound_ids: tf.Tensor) -> tf.Tensor:
    """
    Вывод индексов первого или последнего токена сущностей
    :param label_ids: tf.Tensor of shape [N, T]
    :param bound_ids: tf.Tensor of shape [num_bound_ids] - айдишники, обозначающие начало или конец сущности
    :return: res: tf.Tensor of shape [num_entities_sum, 2], где num_entities_sum - общее число сущностей
             в батче. (i, j) - начало или конец сущности, где 0 <= i < N; 0 < j < T
    TODO: рассмотреть случай неизвестных лейблов токенов. вообще говоря, модель может в качестве первого
     или последнего лейбла сущности предсказать что-то другое (например, I_ORG вместо L_ORG)
    """
    labels_3d = tf.tile(label_ids[:, :, None], [1, 1, tf.shape(bound_ids)[0]])  # [N, T, num_bound_ids]
    mask_3d = tf.equal(labels_3d, bound_ids[None, None, :])  # [N, T, num_bound_ids]
    mask_2d = tf.reduce_any(mask_3d, axis=-1)  # [N, T]
    num_entities = tf.reduce_sum(tf.cast(mask_2d, tf.int32), axis=-1)  # [N]
    sequence_mask = tf.sequence_mask(num_entities)  # [N, num_entities_max]
    indices = tf.cast(tf.where(sequence_mask), tf.int32)  # [num_entities_sum, 2]
    updates = tf.cast(tf.where(mask_2d)[:, -1], tf.int32)  # [num_entities_sum]
    sequence_mask_shape = tf.shape(sequence_mask)
    res = tf.scatter_nd(indices, updates, shape=sequence_mask_shape)  # [N, num_entities_max]

    # Пусть число примеров = 3, число сущностей - 2
    num_examples = sequence_mask_shape[0]
    num_entities_max = sequence_mask_shape[1]
    x = tf.range(num_examples)  # [0, 1, 2]
    x = tf.tile(x[:, None], [1, num_entities_max])  # [[0, 0], [1, 1], [2, 2]]
    x = tf.reshape(x, [-1, 1])  # [[0], [0], [1], [1], [2], [2]]

    y = tf.reshape(res, [-1, 1])
    coords = tf.concat([x, y], axis=-1)
    return coords


def add_ones(x):
    ones = tf.ones_like(x[..., :1])
    x = tf.concat([x, ones], axis=-1)
    return x


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


class SpecialSymbols:
    CLS = '[CLS]'
    SEP = '[SEP]'
    START_HEAD = '[START_HEAD]'
    END_HEAD = '[END_HEAD]'
    START_DEP = '[START_DEP]'
    END_DEP = '[END_DEP]'


class BertEncodings:
    TEXT = "text"
    NER = "ner"
    TEXT_NER = "text_ner"
    NER_TEXT = "ner_text"


class NerEncodings:
    BIO = "bio"
    BILOU = "bilou"
