import pytest
from src.metrics import get_spans, get_ner_metrics, get_f1_precision_recall, f1_score_micro, f1_score_micro_v2


@pytest.mark.parametrize("labels, expected", [
    # нет сущностей
    pytest.param([], {}),
    pytest.param(["O"], {}),
    # одна сущность
    #     1. одно упоминание
    pytest.param(["B-ORG"], {"ORG": {(0, 0)}}),
    pytest.param(["O", "B-ORG"], {"ORG": {(1, 1)}}),
    pytest.param(["O", "B-ORG", "O"], {"ORG": {(1, 1)}}),
    pytest.param(["I-ORG"], {}),
    pytest.param(["O", "I-ORG"], {}),
    pytest.param(["O", "I-ORG", "O"], {}),
    pytest.param(["B-ORG", "I-ORG"], {"ORG": {(0, 1)}}),
    pytest.param(["O", "B-ORG", "I-ORG"], {"ORG": {(1, 2)}}),
    pytest.param(["O", "B-ORG", "I-ORG", "O"], {"ORG": {(1, 2)}}),
    pytest.param(["B-ORG", "O", "I-ORG"], {"ORG": {(0, 0)}}),
    pytest.param(["O", "B-ORG", "O", "I-ORG"], {"ORG": {(1, 1)}}),
    pytest.param(["O", "B-ORG", "O", "I-ORG", "O"], {"ORG": {(1, 1)}}),
    #     2. несколько упоминаний
    pytest.param(["B-ORG", "B-ORG"], {"ORG": {(0, 0), (1, 1)}}),
    pytest.param(["B-ORG", "O", "B-ORG"], {"ORG": {(0, 0), (2, 2)}}),
    pytest.param(["B-ORG", "I-ORG", "B-ORG"], {"ORG": {(0, 1), (2, 2)}}),
    pytest.param(["B-ORG", "I-ORG", "O", "B-ORG"], {"ORG": {(0, 1), (3, 3)}}),
    # несколько сущностей
    pytest.param(["B-ORG", "B-LOC"], {"ORG": {(0, 0)}, "LOC": {(1, 1)}}),
    pytest.param(["B-ORG", "I-LOC"], {"ORG": {(0, 0)}}),
    pytest.param(["B-ORG", "I-ORG", "I-LOC"], {"ORG": {(0, 1)}}),
    pytest.param(["B-ORG", "I-ORG", "B-LOC"], {"ORG": {(0, 1)}, "LOC": {(2, 2)}}),
])
def test_get_spans(labels, expected):
    actual = get_spans(labels)
    assert actual == expected


@pytest.mark.parametrize("tp, fp, fn, expected", [
    pytest.param(0, 0, 0, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}),
    pytest.param(1, 1, 1, {"f1": 0.5, "precision": 0.5, "recall": 0.5, "support": 2})
])
def test_get_f1_precision_recall(tp, fp, fn, expected):
    actual = get_f1_precision_recall(tp=tp, fp=fp, fn=fn)
    assert actual == expected


@pytest.mark.parametrize("y_true, y_pred, expected", [
    pytest.param([], [], {}),
    pytest.param(
        [["B-ORG"], ["B-LOC", "I-LOC"]],
        [["B-ORG"], ["B-LOC", "O"]],
        {
            "ORG": {"f1": 1.0, "precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0, "support": 1},
            "LOC": {"f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 1, "fn": 1, "support": 1},
            "micro": {"f1": 0.5, "precision": 0.5, "recall": 0.5, "tp": 1, "fp": 1, "fn": 1, "support": 2},
        }
    )
])
def test_get_ner_metrics(y_true, y_pred, expected):
    actual = get_ner_metrics(y_true, y_pred)
    assert actual == expected


@pytest.mark.parametrize("y_true, y_pred, expected", [
    pytest.param([], [], {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}),
    pytest.param([0], [0], {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}),
    pytest.param([1], [1], {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}),
    pytest.param([0, 1, 2], [0, 1, 1], {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}),
])
def test_f1_score_micro(y_true, y_pred, expected):
    actual = f1_score_micro(y_true=y_true, y_pred=y_pred, trivial_label=0)
    assert actual == expected, f"{actual} != {expected}"
    actual = f1_score_micro_v2(y_true=y_true, y_pred=y_pred, trivial_label=0)
    assert actual == expected, f"{actual} != {expected}"
