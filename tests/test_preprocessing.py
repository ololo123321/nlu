import pytest
# from collections import namedtuple
# from bert.tokenization import FullTokenizer
from src.data.preprocessing import get_spans


# @pytest.fixture(scope="module")
# def tokenizer():
#     vocab_file = "/home/datascientist/rubert_cased_L-12_H-768_A-12_v2/vocab.txt"
#     return FullTokenizer(vocab_file=vocab_file, do_lower_case=False)


@pytest.mark.parametrize("entity_spans, lengths, window, stride, expected", [
    pytest.param(
        [],
        [1],
        1,
        1,
        [(0, 1)],
        id="no entities"
    ),
    pytest.param(
        [(0, 1)],
        [2],
        1,
        1,
        [(0, 1)],
        id="one entity"
    ),
    pytest.param(
        [(0, 1), (2, 3)],
        [2, 2],
        1,
        1,
        [(0, 1), (1, 2)],
        id="two sentences, one entity in each one"
    ),
    pytest.param(
        [(0, 1), (1, 2)],
        [2, 2],
        1,
        1,
        [(0, 2)],
        id="two sentences, bad split"
    )
])
def test_example_chunks_sentences(entity_spans, lengths, window, stride, expected):
    actual = get_spans(
        entity_spans=entity_spans,
        lengths=lengths,
        window=window,
        stride=stride
    )
    assert actual == expected

# TAG2TOKEN = {
#     "HEAD_PER": "[unused1]",
#     "DEP_PER": "[unused2]",
#     "HEAD_LOC": "[unused3]",
#     "DEP_LOC": "[unused4]",
#     SpecialSymbols.START_HEAD: "[unused5]",
#     SpecialSymbols.START_DEP: "[unused6]",
#     SpecialSymbols.END_HEAD: "[unused7]",
#     SpecialSymbols.END_DEP: "[unused8]",
# }
#
#
# # TODO: тесты других модов
#
# @pytest.mark.parametrize("example, mode, expected", [
#     pytest.param(
#         Example(
#             id="0",
#             tokens=["Мама", "мыла", "раму"],
#             entities=[],
#             arcs=[]
#         ),
#         BertEncodings.NER,
#         []
#     ),
#     pytest.param(
#         Example(
#             id="0",
#             tokens=["Иван", "Иванов", "мыл", "раму"],
#             entities=[
#                 Entity(id="T1", start_token_id=0, end_token_id=1, labels=["B-PER", "L-PER"]),
#             ],
#             arcs=[]
#         ),
#         BertEncodings.NER,
#         []
#     ),
#     pytest.param(
#         Example(
#             id="0",
#             tokens=["Иван", "Иванов", "живёт", "в", "деревне", "Жопа"],
#             entities=[
#                 Entity(id="T1", start_token_id=0, end_token_id=1, labels=["B-PER", "L-PER"]),
#                 Entity(id="T2", start_token_id=5, end_token_id=5, labels=["B-LOC", "L-LOC"]),
#             ],
#             arcs=[]
#         ),
#         BertEncodings.NER,
#         [
#             Example(
#                 id="0_0",
#                 tokens=["[CLS]", "[unused1]", "живёт", "в", "деревне", "[unused4]", "[SEP]"],
#                 label=0
#             ),
#             Example(
#                 id="0_1",
#                 tokens=["[CLS]", "[unused2]", "живёт", "в", "деревне", "[unused3]", "[SEP]"],
#                 label=0
#             )
#         ]
#     ),
#     pytest.param(
#         Example(
#             id="0",
#             tokens=["Иван", "Иванов", "живёт", "в", "деревне", "Жопа"],
#             entities=[
#                 Entity(id="T1", start_token_id=0, end_token_id=1, labels=["B-PER", "L-PER"]),
#                 Entity(id="T2", start_token_id=5, end_token_id=5, labels=["B-LOC", "L-LOC"]),
#             ],
#             arcs=[
#                 Arc(id="R1", head="T1", dep="T2", rel=1)
#             ]
#         ),
#         BertEncodings.NER,
#         [
#             Example(
#                 id="0_0",
#                 tokens=["[CLS]", "[unused1]", "живёт", "в", "деревне", "[unused4]", "[SEP]"],
#                 label=1
#             ),
#             Example(
#                 id="0_1",
#                 tokens=["[CLS]", "[unused2]", "живёт", "в", "деревне", "[unused3]", "[SEP]"],
#                 label=0
#             )
#         ]
#     )
# ])
# def test_convert_example_for_bert(tokenizer, example, mode, expected):
#     actual = convert_example_for_bert(
#         example,
#         tokenizer=tokenizer,
#         tag2token=TAG2TOKEN,
#         mode=mode,
#         no_rel_id=0
#     )
#     for x in actual:
#         x.tokens = tokenizer.convert_ids_to_tokens(x.tokens)
#
#     # print(actual)
#
#     assert len(actual) == len(expected)
#     for x_actual, x_expected in zip(actual, expected):
#         assert x_actual.id == x_expected.id
#         assert x_actual.tokens == x_expected.tokens
#         assert x_actual.label == x_expected.label
