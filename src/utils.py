import random
import re
from collections import defaultdict
from typing import List, Dict, Set

import numpy as np

from src.data.base import Span, Example


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


# TODO: упразднить truncated; сделать так, чтобы колонки выводились в зависиомости от контента d
def classification_report_to_string(d: Dict, digits: int = 4) -> str:
    """
    :param d: словарь вида {"label": {"f1": 0.9, ....}, ...}. см. src.metrics.classification_report
    :param digits: до скольки цифр округлять float
    :return:
    """
    all_metrics = ["f1", "precision", "recall", "support", "tp", "fp", "fn"]

    # check input
    assert len(d) > 0, "empty input"
    input_metrics = list(d.values()).pop().keys()
    for m in input_metrics:
        assert m in all_metrics, f"expected metric to be in {all_metrics}, but got {m}"
    for k, v in d.items():
        assert isinstance(v, dict), f"expected label info to be a dict, but got {v} for label {k}"
        assert v.keys() == input_metrics, f"keys of all labels must be same, but got {v.keys()} != {input_metrics}"

    float_metrics = {"f1", "precision", "recall"}
    col_dist = 2  # расстояние между столбцами
    micro = "micro"
    # так как значения метрик лежат в промежутке [0.0, 1.0], стоит ровно одна цифра слева от точки
    # таким образом, длина числа равна 1 ("0" или "1") + 1 (".") + digits (точность округления)
    max_float_length = digits + 2  # 0.1234

    indices = sorted(d.keys())
    index_length = max(map(len, indices))
    index_length += col_dist

    cols_to_use = [col for col in all_metrics if col in input_metrics]  # для сохранения порядка
    column_length = max(map(len, cols_to_use))
    column_length = max(column_length, max_float_length)
    column_length += col_dist

    report = ' ' * index_length
    for col in cols_to_use:
        report += col.ljust(column_length)
    report += "\n\n"

    def build_row(key):
        row = key.ljust(index_length)
        for metric in cols_to_use:
            if metric in float_metrics:
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


def batches_gen(examples: List[Example], max_tokens_per_batch: int = 10000, pieces_level: bool = False):
    """
    batch_size * max_len_batch <= max_tokens_per_batch
    """
    id2len = {}
    for x in examples:
        if pieces_level:
            id2len[x.id] = sum(len(t.pieces) for t in x.tokens)
        else:
            id2len[x.id] = len(x.tokens)

    examples_sorted = sorted(examples, key=lambda example: id2len[example.id])

    batch = []
    for x in examples_sorted:
        if id2len[x.id] * (len(batch) + 1) <= max_tokens_per_batch:
            batch.append(x)
        else:
            assert len(batch) > 0, f"[{x.id}] too large example: sequence len is {id2len[x.id]}, " \
                f"which is greater than max_tokens_per_batch: {max_tokens_per_batch}"
            yield batch
            batch = [x]
    yield batch


# TODO: разобраться и сделать cythonize
def mst(scores, eps=1e-10):
    """
    Chu-Liu-Edmonds' algorithm for finding minimum spanning arborescence in graphs.
    Calculates the arborescence with node 0 as root.
    :param scores: `scores[i][j]` is the weight of edge from node `j` to node `i`.
    :returns an array containing the head node (node with edge pointing to current node) for each node,
             with head[0] fixed as 0
    1. удалить рёбра вида (*, r)
    2. из пары рёбер ((i, j), (j, i)) выбать ребро с минимальным весом
    3. для каждой вершины child находим вершину parent с минимальным весом
    4. если граф ацикличный, то конец
    5. иначе,
    """
    length = scores.shape[0]
    scores = scores * (1 - np.eye(length))  # mask all the diagonal elements wih a zero
    heads = np.argmax(scores, axis=1)  # THIS MEANS THAT scores[i][j] = score(j -> i)!
    heads[0] = 0  # the root has a self-loop to make it special
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / (head_scores + eps))]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(scores[roots, new_heads] / (root_scores + eps))]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set)  # head -> dep
    vertices = {0}
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / (old_scores + eps)
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)
    return heads


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    _index = [0]
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        _indices[v] = _index[0]
        _lowlinks[v] = _index[0]
        _index[0] += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]
