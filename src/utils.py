import random
from collections import defaultdict
from typing import List, Dict, Set
from src.data.base import Entity, Example, Span


def train_test_split(
    examples: List,
    seed: int = 228,
    train_frac: float = 0.7,
):
    """чтоб не тащить весь sklearn в проект только ради этого"""
    assert train_frac < 1.0

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    num_train = int(len(examples) * train_frac)
    train = [examples[i] for i in indices[:num_train]]
    test = [examples[i] for i in indices[num_train:]]

    return train, test


def train_test_valid_split(
    examples: List,
    seed: int = 228,
    train_frac: float = 0.7,
    test_frac: float = 0.2
):
    assert train_frac + test_frac < 1.0

    train, test_valid = train_test_split(examples, seed=seed, train_frac=train_frac)
    test_frac = test_frac / (1.0 - train_frac)
    test, valid = train_test_split(test_valid, seed=seed, train_frac=test_frac)

    return train, valid, test


def classification_report_to_string(d: Dict, digits: int = 4) -> str:
    """
    :param d: словарь вида {"label": {"f1": 0.9, ....}, ...}. см. src.metrics.classification_report
    :param digits: до скольки цифр округлять float
    :return:
    """
    cols = ["f1", "precision", "recall", "support", "tp", "fp", "fn"]
    float_cols = {"f1", "precision", "recall"}
    col_dist = 2  # расстояние между столбцами
    micro = "micro"
    # так как значения метрик лежат в промежутке [0.0, 1.0], стоит ровно одна цифра слева от точки
    # таким образом длина числа равна 1 ("0" или "1") + 1 (".") + digits (точность округления)
    max_float_length = digits + 2  # 0.1234

    indices = sorted(d.keys())
    index_length = max(map(len, indices))
    index_length += col_dist

    column_length = max(map(len, cols))
    column_length = max(column_length, max_float_length)
    column_length += col_dist

    report = ' ' * index_length
    for col in cols:
        report += col.ljust(column_length)
    report += "\n\n"

    def build_row(key):
        row = key.ljust(index_length)
        for metric in cols:
            if metric in float_cols:
                cell = round(d[key][metric], digits)
            else:
                cell = int(d[key][metric])
            cell = str(cell)
            cell = cell.ljust(column_length)
            row += cell
        return row

    for index in indices:
        if index == micro:
            continue
        r = build_row(index)
        report += r + '\n'

    if micro in indices:
        r = build_row(micro)
        report += "\n" + r
    else:
        report = report.rstrip()

    return report


def get_entity_spans(labels: List[str], joiner: str = '-') -> Dict[str, Set[Span]]:
    """
    поддерживает только кодировку BIO
    :param labels:
    :param joiner:
    :return: map:
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
                tag2spans[entity_tag].add(Span(start, end))
            flag = True
            start = i
            end = i
            entity_tag = tag
        elif bio == "I":
            if flag:
                if tag == entity_tag:
                    end += 1
                else:
                    tag2spans[entity_tag].add(Span(start, end))
                    flag = False
        elif bio == "O":
            if flag:
                tag2spans[entity_tag].add(Span(start, end))
                flag = False
    if flag:
        tag2spans[entity_tag].add(Span(start, end))
    return tag2spans
