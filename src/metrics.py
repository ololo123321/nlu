from typing import List, Dict
from collections import defaultdict


def get_ner_metrics(y_true: List[List[str]], y_pred: List[List[str]], joiner: str = "-") -> Dict:
    assert len(y_true) == len(y_pred)

    d = defaultdict(lambda: defaultdict(int))

    num_examples = len(y_true)
    for i in range(num_examples):
        assert len(y_true[i]) == len(y_pred[i])
        d_true = get_spans(y_true[i], joiner=joiner)
        d_pred = get_spans(y_pred[i], joiner=joiner)
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

    for tag, v in d.items():
        d_tag = get_f1_precision_recall(**v)
        v.update(d_tag)

    return d


def get_spans(labels: List[str], joiner: str = '-') -> Dict:
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


def get_f1_precision_recall(tp: int, fp: int, fn: int) -> Dict:
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


def f1_score_micro(y_true, y_pred, trivial_label: int = 0):
    tp = 0
    fp = 0
    fn = 0
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        if y_true_i != trivial_label or y_pred_i != trivial_label:
            if y_true_i == y_pred_i:
                tp += 1
            elif y_true_i == trivial_label:
                fn += 1
            elif y_pred_i == trivial_label:
                fp += 1

    d = get_f1_precision_recall(tp=tp, fp=fp, fn=fn)
    return d
