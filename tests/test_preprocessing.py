import pytest
from src.preprocessing import Example, Entity, TOKENS_EXPRESSION


@pytest.mark.parametrize("text, tokens, labels, entities, expected", [
    pytest.param(
        "",
        [],
        [],
        [],
        [
            Example(text="", tokens=[], labels=[], entities=[], arcs=[])
        ]
    ),
    pytest.param(
        "Foo. Bar.",
        ["Foo", ".", "Bar", "."],
        ["O", "O", "O", "O"],
        [],
        [
            Example(
                text="Foo."
            ),
            Example(
                text="Bar."
            ),
        ]
    ),
    pytest.param(
        "Foo. Bar.",
        [Entity(id=0, start_token_id=0, end_token_id=3)],
        [
            Example(
                text="Foo. Bar.",
                tokens=["Foo", ".", "Bar", "."],
                labels=["O", ""],
                entities=[Entity(id=0)],
                arcs=[]
            )
        ]
    ),
    pytest.param(
        "Foo. Bar. Baz.",
        [Entity(id=0, start_token_id=0, end_token_id=3)],
        ["Foo. Bar.", "Baz."],
        [[0], []],
        []
    ),
    pytest.param(
        "Foo. Bar. Baz.",
        [Entity(id=0, start_token_id=0, end_token_id=4)],
        ["Foo. Bar. Baz."],
        []
    ),
    pytest.param(
        "Foo. Bar. Baz.",
        [Entity(id=0, start_token_id=0, end_token_id=5)],
        ["Foo. Bar. Baz."],
        []
    ),
    pytest.param(
        "Foo. Bar. Baz.",
        [Entity(id=0, start_token_id=2, end_token_id=5)],
        ["Foo.", "Bar. Baz."],
        []
    ),
    pytest.param(
        "Foo. Bar. Baz.",
        [Entity(id=0, start_token_id=2, end_token_id=3)],
        ["Foo.", "Bar.", "Baz."],
        []
    ),
    pytest.param(
        "Foo. Bar. Baz.",
        [
            Entity(id=0, start_token_id=0, end_token_id=1),
            Entity(id=1, start_token_id=2, end_token_id=5)
        ],
        ["Foo.", "Bar. Baz."],

    ),
])
def test_example_chunks_sentences(text, entities, expected):
    tokens = TOKENS_EXPRESSION.findall(text)
    labels = ["O"] * len(tokens)
    example = Example(text=text, tokens=tokens, labels=labels,  entities=entities)
    chunks = example.chunks
    print(chunks)

    assert len(chunks) == len(expected)

    for x_actual, x_expected in zip(chunks, expected):
        assert x_actual.text == x_expected.text
        assert x_actual.tokens == x_expected.tokens
        assert x_actual.labels == x_expected.labels
        assert {x.id for x in x_actual.entities} == {x.id for x in x_expected.entities}
        assert {x.id for x in x_actual.arcs} == {x.id for x in x_expected.arcs}
