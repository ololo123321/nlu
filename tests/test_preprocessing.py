import pytest
from bert.tokenization import FullTokenizer
from src.preprocessing import Example, Entity, Arc, convert_example_for_bert, BertEncodings, SpecialSymbols


@pytest.fixture(scope="module")
def tokenizer():
    vocab_file = "/home/datascientist/rubert_cased_L-12_H-768_A-12_v2/vocab.txt"
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=False)


@pytest.mark.parametrize("example, expected", [
    pytest.param(
        Example(
            id="0",
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
            id="0",
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
                id="0",
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
            id="0",
            text="Foo. Bar.",
            tokens=["Foo", ".", "Bar", "."],
            labels=["U_ORG", "O", "U_LOC", "O"],
            entities=[],
            arcs=[]
        ),
        [
            Example(
                id="0",
                text="Foo.",
                tokens=["Foo", "."],
                labels=["U_ORG", "O"],
                entities=[],
                arcs=[]
            ),
            Example(
                id="0",
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
            id="0",
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
                id="0",
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
            id="0",
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
                id="0",
                text="Foo. Bar.",
                tokens=["Foo", '.', 'Bar', '.'],
                labels=["B_ORG", "I_ORG", "I_ORG", "I_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=3)
                ],
                arcs=[]
            ),
            Example(
                id="0",
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
            id="0",
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
                id="0",
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
            id="0",
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
                id="0",
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
            id="0",
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
                id="0",
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["O", "O"],
                entities=[],
                arcs=[]
            ),
            Example(
                id="0",
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
            id="0",
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
                id="0",
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["O", "O"],
                entities=[],
                arcs=[]
            ),
            Example(
                id="0",
                text="Bar.",
                tokens=["Bar", '.'],
                labels=["B_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=2, end_token_id=3)
                ],
                arcs=[]
            ),
            Example(
                id="0",
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
            id="0",
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
                id="0",
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["B_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=1)
                ],
                arcs=[]
            ),
            Example(
                id="0",
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
            id="0",
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
                id="0",
                text="Foo.",
                tokens=["Foo", '.'],
                labels=["B_ORG", "L_ORG"],
                entities=[
                    Entity(id=0, start_token_id=0, end_token_id=1)
                ],
                arcs=[]
            ),
            Example(
                id="0",
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
            id="0",
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
                id="0",
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


TAG2TOKEN = {
    "HEAD_PER": "[unused1]",
    "DEP_PER": "[unused2]",
    "HEAD_LOC": "[unused3]",
    "DEP_LOC": "[unused4]",
    SpecialSymbols.START_HEAD: "[unused5]",
    SpecialSymbols.START_DEP: "[unused6]",
    SpecialSymbols.END_HEAD: "[unused7]",
    SpecialSymbols.END_DEP: "[unused8]",
}


# TODO: тесты других модов

@pytest.mark.parametrize("example, mode, expected", [
    pytest.param(
        Example(
            id="0",
            tokens=["Мама", "мыла", "раму"],
            entities=[],
            arcs=[]
        ),
        BertEncodings.NER,
        []
    ),
    pytest.param(
        Example(
            id="0",
            tokens=["Иван", "Иванов", "мыл", "раму"],
            entities=[
                Entity(id="T1", start_token_id=0, end_token_id=1, labels=["B-PER", "L-PER"]),
            ],
            arcs=[]
        ),
        BertEncodings.NER,
        []
    ),
    pytest.param(
        Example(
            id="0",
            tokens=["Иван", "Иванов", "живёт", "в", "деревне", "Жопа"],
            entities=[
                Entity(id="T1", start_token_id=0, end_token_id=1, labels=["B-PER", "L-PER"]),
                Entity(id="T2", start_token_id=5, end_token_id=5, labels=["B-LOC", "L-LOC"]),
            ],
            arcs=[]
        ),
        BertEncodings.NER,
        [
            Example(
                id="0_0",
                tokens=["[CLS]", "[unused1]", "живёт", "в", "деревне", "[unused4]", "[SEP]"],
                label=0
            ),
            Example(
                id="0_1",
                tokens=["[CLS]", "[unused2]", "живёт", "в", "деревне", "[unused3]", "[SEP]"],
                label=0
            )
        ]
    ),
    pytest.param(
        Example(
            id="0",
            tokens=["Иван", "Иванов", "живёт", "в", "деревне", "Жопа"],
            entities=[
                Entity(id="T1", start_token_id=0, end_token_id=1, labels=["B-PER", "L-PER"]),
                Entity(id="T2", start_token_id=5, end_token_id=5, labels=["B-LOC", "L-LOC"]),
            ],
            arcs=[
                Arc(id="R1", head="T1", dep="T2", rel=1)
            ]
        ),
        BertEncodings.NER,
        [
            Example(
                id="0_0",
                tokens=["[CLS]", "[unused1]", "живёт", "в", "деревне", "[unused4]", "[SEP]"],
                label=1
            ),
            Example(
                id="0_1",
                tokens=["[CLS]", "[unused2]", "живёт", "в", "деревне", "[unused3]", "[SEP]"],
                label=0
            )
        ]
    )
])
def test_convert_example_for_bert(tokenizer, example, mode, expected):
    actual = convert_example_for_bert(
        example,
        tokenizer=tokenizer,
        tag2token=TAG2TOKEN,
        mode=mode,
        no_rel_id=0
    )
    for x in actual:
        x.tokens = tokenizer.convert_ids_to_tokens(x.tokens)

    # print(actual)

    assert len(actual) == len(expected)
    for x_actual, x_expected in zip(actual, expected):
        assert x_actual.id == x_expected.id
        assert x_actual.tokens == x_expected.tokens
        assert x_actual.label == x_expected.label
