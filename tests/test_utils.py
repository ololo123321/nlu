import pytest
from src.utils import get_entity_spans, get_connected_components


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
    actual = get_entity_spans(labels)
    assert actual == expected


@pytest.mark.parametrize("g, expected", [
    pytest.param({}, set()),
    pytest.param({1: set(), 2: set()}, {frozenset({1}), frozenset({2})}),
    pytest.param({1: {2}, 2: {1}, 3: set()}, {frozenset({1, 2}), frozenset({3})}),
    pytest.param({1: set(), 2: {1}, 3: set()}, {frozenset({1, 2}), frozenset({3})})
])
def test_get_connected_components(g, expected):
    components = get_connected_components(g)
    actual = set(map(frozenset, components))
    assert actual == expected


def test_get_connected_components_error():
    g = {1: {2}}
    with pytest.raises(AssertionError):
        get_connected_components(g)
