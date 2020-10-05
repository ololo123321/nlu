import pytest
from src.preprocessing import Example, Entity, Arc


@pytest.mark.parametrize("example, expected", [
    pytest.param(
        Example(
            text="",
            tokens=[],
            labels=[],
            entities=[],
            arcs=[]
        ),
        [
            Example(text="", tokens=[], labels=[], entities=[], arcs=[])
        ],
        id="0"
    ),
    pytest.param(
        Example(
            text="foo",
            tokens=["foo"],
            labels=["O"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=0)
            ],
            arcs=[]
        ),
        [
            Example(
                text="foo",
                tokens=["foo"],
                labels=["O"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=0)
                ],
                arcs=[]
            )
        ],
        id="00"
    ),
    pytest.param(
        Example(
            text="Foo. Bar.",
            tokens=["Foo", ".", "Bar", "."],
            labels=["U_ORG", "O", "U_LOC", "O"],
            entities=[],
            arcs=[]
        ),
        [
            Example(
                text="Foo.",
                tokens=["Foo", "."],
                labels=["U_ORG", "O"],
                entities=[],
                arcs=[]
            ),
            Example(
                text="Bar.",
                tokens=["Bar", "."],
                labels=["U_LOC", "O"],
                entities=[],
                arcs=[]
            ),
        ],
        id="1"
    ),
    pytest.param(
        Example(
            text="Foo. Bar.",
            tokens=["Foo", ".", "Bar", "."],
            labels=["B_ORG", "I_ORG", "I_ORG", "L_ORG"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=3)
            ],
            arcs=[]
        ),
        [
            Example(
                text="Foo. Bar.",
                tokens=["Foo", ".", "Bar", "."],
                labels=["B_ORG", "I_ORG", "I_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=3)
                ],
                arcs=[]
            ),
        ],
        id="2"
    ),
    pytest.param(
        Example(
            text="Foo. Bar. Baz.",
            tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
            labels=["B_ORG", "I_ORG", "I_ORG", "I_ORG", "O", "O"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=3)
            ],
            arcs=[]
        ),
        [
            Example(
                text="Foo. Bar.",
                tokens=["Foo", '.', 'Bar', '.'],
                labels=["B_ORG", "I_ORG", "I_ORG", "I_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=3)
                ],
                arcs=[]
            ),
            Example(
                text="Baz.",
                tokens=['Baz', '.'],
                labels=["O", "O"],
                entities=[],
                arcs=[]
            )
        ],
        id="3"
    ),
    pytest.param(
        Example(
            text="Foo. Bar. Baz.",
            tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
            labels=["B_ORG", "I_ORG", "I_ORG", "I_ORG", "O", "O"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=4)
            ],
            arcs=[]
        ),
        [
            Example(
                text="Foo. Bar. Baz.",
                tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
                labels=["B_ORG", "I_ORG", "I_ORG", "I_ORG", "O", "O"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=4)
                ],
                arcs=[]
            )
        ],
        id="4"
    ),
    pytest.param(
        Example(
            text="Foo. Bar. Baz.",
            tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
            labels=["B_ORG", "I_ORG", "I_ORG", "I_ORG", "I_ORG", "L_ORG"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=5)
            ],
            arcs=[]
        ),
        [
            Example(
                text="Foo. Bar. Baz.",
                tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
                labels=["B_ORG", "I_ORG", "I_ORG", "I_ORG", "I_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=5)
                ],
                arcs=[]
            )
        ],
        id="5"
    ),
    pytest.param(
        Example(
            text="Foo. Bar. Baz.",
            tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
            labels=["O", "O", "B_ORG", "I_ORG", "I_ORG", "L_ORG"],
            entities=[
                Entity(id=0, start_token_id=2, end_token_id=5)
            ],
            arcs=[]
        ),
        [
            Example(
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["O", "O"],
                entities=[],
                arcs=[]
            ),
            Example(
                text="Bar. Baz.",
                tokens=['Bar', '.', 'Baz', '.'],
                labels=["B_ORG", "I_ORG", "I_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=2, end_token_id=5)
                ],
                arcs=[]
            ),
        ],
        id="6"
    ),
    pytest.param(
        Example(
            text="Foo. Bar. Baz.",
            tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
            labels=["O", "O", "B_ORG", "L_ORG", "B_ORG", "L_ORG"],
            entities=[
                Entity(id=0, start_token_id=2, end_token_id=3),
                Entity(id=1, start_token_id=4, end_token_id=5)
            ],
            arcs=[]
        ),
        [
            Example(
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["O", "O"],
                entities=[],
                arcs=[]
            ),
            Example(
                text="Bar.",
                tokens=["Bar", '.'],
                labels=["B_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=2, end_token_id=3)
                ],
                arcs=[]
            ),
            Example(
                text="Baz.",
                tokens=["Baz", '.'],
                labels=["B_ORG", "L_ORG"],
                entities=[
                    Entity(id=1, start_token_id=4, end_token_id=5)
                ],
                arcs=[]
            ),
        ],
        id="7"
    ),
    pytest.param(
        Example(
            text="Foo. Bar. Baz.",
            tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
            labels=["B_ORG", "L_ORG", "B_ORG", "I_ORG", "L_ORG", "O"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=1),
                Entity(id=1, start_token_id=2, end_token_id=4)
            ],
            arcs=[]
        ),
        [
            Example(
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["B_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=1)
                ],
                arcs=[]
            ),
            Example(
                text="Bar. Baz.",
                tokens=['Bar', '.', 'Baz', '.'],
                labels=["B_ORG", "I_ORG", "L_ORG", "O"],
                entities=[
                    Entity(id=1, start_token_id=4, end_token_id=3)
                ],
                arcs=[]
            )
        ],
        id="8"
    ),
    pytest.param(
        Example(
            text="Foo. Bar. Baz.",
            tokens=["Foo", '.', 'Bar', '.', 'Baz', '.'],
            labels=["B_ORG", "L_ORG", "B_ORG", "I_ORG", "L_ORG", "O"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=1),
                Entity(id=1, start_token_id=2, end_token_id=4)
            ],
            arcs=[Arc(id=0, head=0, dep=1, rel=0)]
        ),
        [
            Example(
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["B_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=1)
                ],
                arcs=[]
            ),
            Example(
                text="Bar. Baz.",
                tokens=['Bar', '.', 'Baz', '.'],
                labels=["B_ORG", "I_ORG", "L_ORG", "O"],
                entities=[
                    Entity(id=1, start_token_id=4, end_token_id=3)
                ],
                arcs=[]
            )
        ],
        id="9; relation between entities in different sentences"
    ),
    pytest.param(
        Example(
            text="Foo Bar",
            tokens=["Foo", 'Bar'],
            labels=["U_ORG", "U_PER"],
            entities=[
                Entity(id=0, start_token_id=0, end_token_id=0),
                Entity(id=1, start_token_id=1, end_token_id=1)
            ],
            arcs=[Arc(id=0, head=0, dep=1, rel=0)]
        ),
        [
            Example(
                text="Foo Bar",
                tokens=["Foo", 'Bar'],
                labels=["U_ORG", "U_PER"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=0),
                    Entity(id=1, start_token_id=1, end_token_id=1)
                ],
                arcs=[Arc(id=0, head=0, dep=1, rel=0)]
            ),
        ],
        id="10; relation between entities in one sentence"
    )
])
def test_example_chunks_sentences(example, expected):
    chunks = example.chunks
    # print(chunks)

    assert len(chunks) == len(expected)

    for x_actual, x_expected in zip(chunks, expected):
        assert x_actual.text == x_expected.text
        assert x_actual.tokens == x_expected.tokens
        assert x_actual.labels == x_expected.labels
        assert {x.id for x in x_actual.entities} == {x.id for x in x_expected.entities}
        assert {x.id for x in x_actual.arcs} == {x.id for x in x_expected.arcs}
