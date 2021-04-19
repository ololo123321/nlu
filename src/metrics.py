import subprocess
from typing import List, Dict, Union, Set, Tuple
from collections import defaultdict
from src.utils import get_entity_spans


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
                    # y_true_i = 0, y_pred_i = 0
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


def get_coreferense_resolution_metrics(path_true, path_pred, scorer_path, metric: str = "all"):
    valid_metrics = {"all", "muc", "bcub", "ceafm", "ceafe", "blanc"}
    assert metric in valid_metrics, f"expected metric in {valid_metrics}, but got {metric}"
    cmd = ["perl", scorer_path, metric, path_true, path_pred, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    stdout = stdout.decode("utf-8")
    if stderr is not None:
        print("captured stderr:")
        print(stderr)
    return stdout


def classification_report_set(y_true: Set[Tuple], y_pred: Set[Tuple]) -> Dict:
    """
    ребро - (head_label, start_head, end_head, dep_label, start_dep, end_dep, relation_label)
    :param y_true:
    :param y_pred:
    :return:
    TODO: учесть то, что если head или dep являются триггером события,
     то не критично неверное определение индексов start и end
    """
    tp = len(y_true & y_pred)
    fp = len(y_pred) - tp
    fn = len(y_true) - tp
    return f1_precision_recall_support(tp=tp, fp=fp, fn=fn)
