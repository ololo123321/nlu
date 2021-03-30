import pytest
from src.data.base import Languages, Example, Token, Span
from src.data.preprocessing import get_sentences_spans, split_example_v1, split_example_v2, apply_bpe


# get spans

# TODO: лучше покрыть тестами случаи (w=2, s=1); (w=2, s=2)
@pytest.mark.parametrize("entity_spans, pointers, window, stride, expected", [
    # w=1, s=1
    pytest.param([], [], 1, 1, [], id="no pointers"),
    pytest.param([], [0, 1], 1, 1, [(0, 1)], id="no entities"),
    pytest.param([(0, 1)], [0, 2], 1, 1, [(0, 1)], id="one entity"),
    pytest.param([(0, 1), (2, 3)], [0, 2, 4], 1, 1, [(0, 1), (1, 2)], id="two sentences, one entity in each one"),
    pytest.param([(0, 1), (1, 2)], [0, 2, 4], 1, 1, [(0, 2)], id="two sentences, bad split"),
    pytest.param([(0, 1), (1, 2), (4, 5)], [0, 2, 4, 6], 1, 1, [(0, 2), (2, 3)],
                 id="three sentences, 1st split is bad"),
    pytest.param([(2, 3), (3, 4), (5, 6)], [0, 2, 4, 6], 1, 1, [(0, 1), (1, 3)],
                 id="three sentences, 2nd split is bad"),
    pytest.param([(1, 2), (3, 4)], [0, 2, 4, 6], 1, 1, [(0, 3)],
                 id="three sentences, all splits are bad"),
    # w=2, s=1
    pytest.param([(0, 1), (2, 3)], [0, 2, 4], 2, 1, [(0, 2)]),
    pytest.param([(0, 1), (2, 3), (4, 5)], [0, 2, 4, 6], 2, 1, [(0, 2), (1, 3)]),
    pytest.param([(0, 1), (2, 5)], [0, 2, 4, 6], 2, 1, [(0, 3)]),
    # w=2, s=2
    pytest.param([(0, 1), (2, 3), (4, 5)], [0, 2, 4, 6], 2, 2, [(0, 2), (2, 3)]),
])
def test_get_sentences_spans(entity_spans, pointers, window, stride, expected):
    actual = get_sentences_spans(
        entity_spans=entity_spans,
        pointers=pointers,
        window=window,
        stride=stride
    )
    # print(actual)
    assert actual == expected


def test_get_sentences_spans_raise():
    # pointers состоит из одного элемента
    with pytest.raises(AssertionError):
        get_sentences_spans(entity_spans=[], pointers=[1])

    # первый элемент pointers не ноль
    with pytest.raises(AssertionError):
        get_sentences_spans(entity_spans=[], pointers=[1, 2])

    # stride > window
    with pytest.raises(AssertionError):
        get_sentences_spans(entity_spans=[], pointers=[], stride=2, window=1)


# split example


def _test_split_example(split_fn, example, window, stride, expected_num_chunks):
    chunks = split_fn(example=example, window=window, stride=stride, lang=Languages.RU)
    assert len(chunks) == expected_num_chunks
    for chunk in chunks:
        for t in chunk.tokens:
            expected = t.text
            actual = chunk.text[t.span_rel[0]:t.span_rel[1]]
            assert actual == expected, f"{actual} != {expected}"
            actual = example.text[t.span_abs[0]:t.span_abs[1]]
            assert actual == expected, f"{actual} != {expected}"
        for entity in chunk.entities:
            for t_link in entity.tokens:
                t_original = chunk.tokens[t_link.index_rel]
                assert id(t_link) == id(t_original)


# TODO: рассмотреть случай, при котором в исходном примере есть сущности, отношения
@pytest.mark.parametrize("example, window, stride, expected_num_chunks", [
    # no entities
    pytest.param(
        Example(
            id="0",
            text="Мама мыла раму.",
            tokens=[
                Token(text="Мама", span_abs=Span(0, 4)),
                Token(text="мыла", span_abs=Span(5, 9)),
                Token(text="раму", span_abs=Span(10, 14)),
                Token(text=".", span_abs=Span(14, 15)),
            ]
        ), 1, 1, 1,
        id="one chunk, no entities"
    ),
    pytest.param(
        Example(
            id="0",
            text="Мама мыла раму. Мама - молодец.",
            tokens=[
                Token(text="Мама", span_abs=Span(0, 4)),
                Token(text="мыла", span_abs=Span(5, 9)),
                Token(text="раму", span_abs=Span(10, 14)),
                Token(text=".", span_abs=Span(14, 15)),
                Token(text="Мама", span_abs=Span(16, 20)),
                Token(text="-", span_abs=Span(21, 22)),
                Token(text="молодец", span_abs=Span(23, 30)),
                Token(text=".", span_abs=Span(30, 31))
            ]
        ), 1, 1, 2,
        id="two chunks, no entities"
    )
])
def test_split_example(example, window, stride, expected_num_chunks):
    _test_split_example(split_example_v1, example, window, stride, expected_num_chunks)
    _test_split_example(split_example_v2, example, window, stride, expected_num_chunks)


# apply bpe


class Tokenizer:
    vocab = {
        "мама": ["мама"],
        "мыла": ["мы", "#ла"],
        "раму": ["ра", "#м", "#у"]
    }
    @classmethod
    def tokenize(cls, text: str):
        return cls.vocab[text]

    @staticmethod
    def convert_tokens_to_ids(tokens):
        return [0] * len(tokens)


@pytest.mark.parametrize("example, expected_pieces, expected_labels", [
    pytest.param(
        Example(
            tokens=[
                Token(text="мама", labels=["B-PER"]),
                Token(text="мыла", labels=["I-PER"]),
                Token(text="раму", labels=["O"])
            ]
        ),
        ["мама", "мы", "#ла", "ра", "#м", "#у"],
        ["B-PER", "I-PER", "I-PER", "O", "O", "O"]
    )
])
def test_apply_bpe(example, expected_pieces, expected_labels):
    apply_bpe(example=example, tokenizer=Tokenizer, ner_prefix_joiner="-", ner_encoding="bio")
    actual_pieces = []
    actual_labels = []
    for t in example.tokens:
        actual_pieces += t.pieces
        actual_labels += t.labels_pieces
    assert actual_pieces == expected_pieces
    assert actual_labels == expected_labels
