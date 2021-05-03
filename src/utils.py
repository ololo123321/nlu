import random
import re
from collections import defaultdict
from typing import List, Dict, Set
from src.data.base import Span


def train_test_split(
    examples: List,
    train_frac: float = 0.7,
    seed: int = 228,
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


def classification_report_to_string(d: Dict, digits: int = 4, truncated: bool = False) -> str:
    """
    :param d: словарь вида {"label": {"f1": 0.9, ....}, ...}. см. src.metrics.classification_report
    :param digits: до скольки цифр округлять float
    :param truncated: если True, то выводить только f1, precision, recall
    :return:
    """
    cols = ["f1", "precision", "recall"]
    if not truncated:
        cols += ["support", "tp", "fp", "fn"]

    # check input
    for k, v in d.items():
        assert isinstance(v, dict), f"expected label info to be a dict with keys {cols}, but got {v} for label {k}"
        for col in cols:
            assert col in v.keys(), f"label {k} has no key {col}"

    float_cols = {"f1", "precision", "recall"}
    col_dist = 2  # расстояние между столбцами
    micro = "micro"
    # так как значения метрик лежат в промежутке [0.0, 1.0], стоит ровно одна цифра слева от точки
    # таким образом, длина числа равна 1 ("0" или "1") + 1 (".") + digits (точность округления)
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
        else:
            raise NotImplementedError(f"only BIO encoding supported, but got label {label}")

    if flag:
        tag2spans[entity_tag].add(Span(start, end))
    return tag2spans


def get_connected_components(g: Dict) -> List:
    """
    {1: set(), 2: {1}, 3: set()} -> [[1, 2], [3]]
    g - граф в виде родитель -> дети
    если среди детей есть такой, что его нет в множестве ключей g, то вызвать ошибку
    :param g:
    :return:
    """
    vertices = set()
    g2 = defaultdict(set)
    for parent, children in g.items():
        vertices.add(parent)
        for child in children:
            assert child in g, f"unknown node {child} among children of {parent}"
            g2[parent].add(child)
            g2[child].add(parent)
    components = []
    while vertices:
        root = vertices.pop()
        comp = dfs(g2, root, warn_on_cycles=False)
        components.append(comp)
        for v in comp:
            if v != root:
                vertices.remove(v)
    return components


def get_strongly_connected_components(g: Dict) -> List:
    """
    {1: set(), 2: {1}, 3: set()} -> [[1], [2], [3]]
    пока не надо
    """


def dfs(g: Dict[str, Set[str]], v: str, warn_on_cycles: bool = False):
    visited = set()

    def traverse(i):
        visited.add(i)
        for child in g[i]:
            if child not in visited:
                traverse(child)
            else:
                if warn_on_cycles:
                    print(f"graph contains cycles: last edge is {i} -> {child}")

    traverse(v)
    return visited


_coref_pattern = r".*{}: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*"
COREF_RESULTS_REGEX = re.compile(_coref_pattern.format("Coreference"), re.DOTALL)
COREF_RESULTS_REGEX_BLANC = re.compile(_coref_pattern.format("BLANC"), re.DOTALL)


def parse_conll_metrics(stdout: str, is_blanc: bool) -> Dict:
    expression = COREF_RESULTS_REGEX_BLANC if is_blanc else COREF_RESULTS_REGEX
    coref_results_match = expression.match(stdout)
    d = {
        "recall": float(coref_results_match.group(1)) * 0.01,
        "precision": float(coref_results_match.group(2)) * 0.01,
        "f1": float(coref_results_match.group(3)) * 0.01
    }
    return d
