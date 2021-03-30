from typing import List, Dict, Union
from collections import defaultdict


def classification_report(
        y_true: List[Union[int, str]],
        y_pred: List[Union[int, str]],
        trivial_label: Union[int, str] = 0
) -> Dict:
    """
    {
        "label_1": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 10, "tp": 10, "fp": 0, "fn": 0},
        ...
        "label_n": ...,
        "micro": ...
    }
    """
    assert len(y_true) == len(y_pred)
    d = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    d["micro"] = d["micro"]  # обязательный ключ

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] != trivial_label:
                d[y_true[i]]["tp"] += 1
                d["micro"]["tp"] += 1
        else:
            if y_true[i] == trivial_label:
                if y_pred[i] == trivial_label:
                    pass
                else:
                    # y_true_i = 0, y_pred_i = 2
                    d[y_pred[i]]["fp"] += 1
                    d["micro"]["fp"] += 1
            else:
                if y_pred[i] == trivial_label:
                    # y_true_i = 2, y_pred_i = 0
                    d[y_true[i]]["fn"] += 1
                    d["micro"]["fn"] += 1
                else:
                    # y_true_i = 2, y_pred_i = 1
                    d[y_true[i]]["fn"] += 1
                    d[y_pred[i]]["fp"] += 1
                    d["micro"]["fn"] += 1
                    d["micro"]["fp"] += 1

    for v in d.values():
        d_tag = f1_precision_recall_support(**v)
        v.update(d_tag)

    return d


def classification_report_ner(y_true: List[List[str]], y_pred: List[List[str]], joiner: str = "-") -> Dict:
    """
    тот же формат, что и classification_report
    """
    assert len(y_true) == len(y_pred)
    d = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    d["micro"] = d["micro"]  # обязательный ключ

    for i in range(len(y_true)):
        assert len(y_true[i]) == len(y_pred[i])
        d_true = get_entity_spans(y_true[i], joiner=joiner)
        d_pred = get_entity_spans(y_pred[i], joiner=joiner)
        common_tags = set(d_true.keys()) | set(d_pred.keys())
        for tag in common_tags:
            tp = len(d_true[tag] & d_pred[tag])
            fp = len(d_pred[tag]) - tp
            fn = len(d_true[tag]) - tp
            d[tag]["tp"] += tp
            d[tag]["fp"] += fp
            d[tag]["fn"] += fn
            d["micro"]["tp"] += tp
            d["micro"]["fp"] += fp
            d["micro"]["fn"] += fn

    for v in d.values():
        d_tag = f1_precision_recall_support(**v)
        v.update(d_tag)

    return d


def get_entity_spans(labels: List[str], joiner: str = '-') -> Dict:
    """
    поддерживает только кодировку BIO
    :param labels:
    :param joiner:
    :return:
    """
    tag2spans = defaultdict(set)

    num_labels = len(labels)
    entity_tag = None
    start = 0
    end = 0
    # поднятие:
    # 1. B-*
    # опускание:
    # 1. O
    # 2. I-{другой таг}

    flag = False

    for i in range(num_labels):
        label = labels[i]
        bio = label[0]
        tag = label.split(joiner)[-1]
        if bio == "B":
            if entity_tag is not None:
                tag2spans[entity_tag].add((start, end))
            flag = True
            start = i
            end = i
            entity_tag = tag
        elif bio == "I":
            if flag:
                if tag == entity_tag:
                    end += 1
                else:
                    tag2spans[entity_tag].add((start, end))
                    flag = False
        elif bio == "O":
            if flag:
                tag2spans[entity_tag].add((start, end))
                flag = False
    if flag:
        tag2spans[entity_tag].add((start, end))
    return tag2spans


def f1_precision_recall_support(tp: int, fp: int, fn: int) -> Dict:
    pos_pred = tp + fp
    if pos_pred == 0:
        precision = 0.0
    else:
        precision = tp / pos_pred

    support = tp + fn
    if support == 0:
        recall = 0.0
    else:
        recall = tp / support

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    d = {"f1": f1, "precision": precision, "recall": recall, "support": support}

    return d


def _f1_score_micro_v2(y_true: List, y_pred: List, trivial_label: Union[int, str] = 0):
    """
    Альтернативная реализация f1_score_micro, для подстраховки.
    """
    assert len(y_true) == len(y_pred)
    tp = 0
    num_pred = 0
    num_gold = 0
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        if y_true_i != trivial_label:
            num_gold += 1
        if y_pred_i != trivial_label:
            num_pred += 1
        if (y_true_i == y_pred_i) and (y_true_i != trivial_label) and (y_pred_i != trivial_label):
            tp += 1

    if num_pred == 0:
        precision = 0.0
    else:
        precision = tp / num_pred

    if num_gold == 0:
        recall = 0.0
    else:
        recall = tp / num_gold

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    d = {"f1": f1, "precision": precision, "recall": recall, "support": num_gold}

    return d
