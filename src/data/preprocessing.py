from copy import deepcopy
from typing import List, Pattern, Tuple, Dict
from itertools import accumulate
from rusenttokenize import ru_sent_tokenize
from collections import defaultdict
import nltk

from .base import (
    Entity,
    Example,
    Languages,
    NerEncodings,
    NerPrefixJoiners,
    TOKENS_EXPRESSION,
    Token,
    Span,
)

# split


def split_example_v1(
        example: Example,
        window: int = 1,
        stride: int = 1,
        lang: str = Languages.RU,
        tokens_expression: Pattern = None
) -> List[Example]:
    """
    Кусок исходного примера размером window предложений
    Реализовано через deepcopy куска исходного примера, что ОЧЕНЬ дорого: узкое место: deepcopy слайса токенов

    :param example: пример на уровне документа
    :param window: ширина окна на уровне предложений
    :param stride: страйд
    :param lang: язык
    :param tokens_expression
    :return:
    """
    assert example.id is not None

    if not example.text:
        print(f"[{example.id} WARNING]: empty text")
        return [Example(**example.__dict__)]

    if lang == Languages.RU:
        split_fn = ru_sent_tokenize
    else:
        split_fn = nltk.sent_tokenize

    if tokens_expression is not None:
        expression = tokens_expression
    else:
        expression = TOKENS_EXPRESSION

    sent_candidates = [sent for sent in split_fn(example.text) if len(sent) > 0]
    lengths = [len(expression.findall(sent)) for sent in sent_candidates]
    assert sum(lengths) == len(example.tokens)

    pointers = [0] + list(accumulate(lengths))
    entity_spans = [
        Span(start=entity.tokens[0].index_abs, end=entity.tokens[-1].index_abs) for entity in example.entities
    ]
    sent_spans = get_sentences_spans(
        entity_spans=entity_spans,
        pointers=pointers,
        window=window,
        stride=stride
    )

    res = []
    for i, span in enumerate(sent_spans):
        text = ' '.join(sent_candidates[span.start:span.end])
        start = pointers[span.start]
        end = pointers[span.end]

        # deepcopy - медленная штука
        example_copy = deepcopy(Example(
            filename=example.filename,
            id=f"{example.id}_{i}",
            text=text,
            tokens=example.tokens[start:end],
            entities=example.entities,
            events=example.events,
            arcs=example.arcs,
            label=example.label
        ))

        # tokens
        # TODO: рассмотреть случай, при котором text начинается с пробелов
        offset = example_copy.tokens[0].span_abs.start
        for j, t in enumerate(example_copy.tokens):
            t.span_rel = Span(start=t.span_abs.start - offset, end=t.span_abs.end - offset)
            t.index_rel = j

        # entities
        entity_ids = set()
        entities = []
        for entity in example_copy.entities:
            if start <= entity.tokens[0].index_abs <= entity.tokens[-1].index_abs < end:
                entities.append(entity)
                entity_ids.add(entity.id)
        example_copy.entities = entities

        # events
        example_copy.events = [event for event in example_copy.events if event.trigger in entity_ids]

        # arcs
        example_copy.arcs = [arc for arc in example_copy.arcs if (arc.head in entity_ids) and (arc.dep in entity_ids)]

        res.append(example_copy)

    return res


def split_example_v2(
        example: Example,
        window: int = 1,
        stride: int = 1,
        lang: str = Languages.RU,
        tokens_expression: Pattern = None
) -> List[Example]:
    """
    Кусок исходного примера размером window предложений.
    deepcopy применяется только к копированию инстансов класса Arc, Event.
    работает на порядок быстрее, чем v1

    :param example: пример на уровне документа
    :param window: ширина окна на уровне предложений
    :param stride: страйд
    :param lang: язык
    :param tokens_expression:
    :return:
    """
    if not example.text:
        print(f"[{example.id} WARNING]: empty text")
        return [Example(**example.__dict__)]

    split_fn = ru_sent_tokenize if lang == Languages.RU else nltk.sent_tokenize
    expression = tokens_expression if tokens_expression is not None else TOKENS_EXPRESSION

    sent_candidates = [sent for sent in split_fn(example.text) if len(sent) > 0]
    lengths = [len(expression.findall(sent)) for sent in sent_candidates]
    assert sum(lengths) == len(example.tokens)

    pointers = [0] + list(accumulate(lengths))
    entity_spans = [
        Span(start=entity.tokens[0].index_abs, end=entity.tokens[-1].index_abs) for entity in example.entities
    ]
    sent_spans = get_sentences_spans(
        entity_spans=entity_spans,
        pointers=pointers,
        window=window,
        stride=stride
    )

    # print("spans_sents:", spans_sents)
    # print("spans_tokens:", spans_tokens)

    res = []
    for i, span in enumerate(sent_spans):
        text = ' '.join(sent_candidates[span.start:span.end])
        start = pointers[span.start]
        end = pointers[span.end]

        # tokens
        # TODO: рассмотреть случай, при котором text начинается с пробелов
        # tokens = example.tokens[span.span_tokens[0]:span.span_tokens[1]]
        tokens = []
        # print(len(example.tokens), start_token)
        offset = example.tokens[start].span_abs.start
        token_assignment_map = {}
        for j, t in enumerate(example.tokens[start:end]):
            t_copy = Token(
                text=t.text,
                span_abs=t.span_abs,
                span_rel=Span(start=t.span_abs.start - offset, end=t.span_abs.end - offset),
                index_abs=t.index_abs,
                index_rel=j,
                labels=t.labels.copy(),
                pieces=t.pieces.copy()
            )
            tokens.append(t_copy)
            token_assignment_map[t] = t_copy

        # entities
        entity_ids = set()
        entities = []
        for entity in example.entities:
            if start <= entity.tokens[0].index_abs <= entity.tokens[-1].index_abs < end:
                entity_new = Entity(
                    id=entity.id,
                    label=entity.label,
                    text=entity.text,
                    tokens=[token_assignment_map[t] for t in entity.tokens],
                    is_event_trigger=entity.is_event_trigger,
                    attrs=entity.attrs.copy(),
                    comment=entity.comment
                )
                entities.append(entity_new)
                entity_ids.add(entity.id)

        # events  TODO: сделать без deepcopy
        events = [deepcopy(event) for event in example.events if event.trigger in entity_ids]

        # arcs  TODO: сделать без deepcopy
        arcs = [deepcopy(arc) for arc in example.arcs if (arc.head in entity_ids) and (arc.dep in entity_ids)]

        example_copy = Example(
            filename=example.filename,
            id=f"{example.id}_{i}",
            text=text,
            tokens=tokens,
            entities=entities,
            events=events,
            arcs=arcs,
            label=example.label
        )

        res.append(example_copy)

    return res


def get_sentences_spans(entity_spans: List[Span], pointers: List[int], window: int = 1, stride: int = None) -> List[Span]:
    """

    :param entity_spans: индексы токенов границ именных сущностей
    :param pointers: индексы токенов, определяющий границы предложений.
    :param window:
    :param stride:
    :return: spans: список спанов предложений
    """
    # stride
    if stride is None:
        stride = window
    else:
        assert stride <= window

    # pointers
    if len(pointers) == 0:
        return []
    elif len(pointers) == 1:
        raise AssertionError
    else:
        assert pointers[0] == 0

    num_sentences = len(pointers) - 1
    if window >= num_sentences:
        return [Span(start=0, end=num_sentences)]

    res = []
    start = 0
    end = window
    is_good_split = True  # разделение на предложения плохое, если оно проходит через именную сущность
    span_ptr = 0
    entity_spans_sorted = sorted(entity_spans)
    while True:
        # print(span_ptr)
        end_token_id = pointers[end]

        # провекра разделения на предложения
        for span in entity_spans_sorted[span_ptr:]:
            if span[0] < end_token_id:
                span_ptr += 1
                if span[1] >= end_token_id:
                    is_good_split = False
                    break
            else:
                break
        # for span in entity_spans:
        #     if span[0] < end_token_id <= span[1]:
        #         is_good_split = False
        #         break

        if is_good_split:
            res.append(Span(start=start, end=end))
            start = min(num_sentences - 1, start + stride)
            end = min(num_sentences, start + window)
        else:
            end = min(num_sentences, end + 1)

        if end == num_sentences:
            res.append(Span(start=start, end=end))
            break

        # присвоение флагу is_good_split дефолтного значения
        is_good_split = True

    return res


def apply_bpe(
        example: Example,
        tokenizer,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        ner_encoding: str = NerEncodings.BIO
):
    if ner_encoding != NerEncodings.BIO:
        raise NotImplementedError

    for t in example.tokens:
        t.pieces = tokenizer.tokenize(t.text)
        t.token_ids = tokenizer.convert_tokens_to_ids(t.pieces)
        num_pieces = len(t.pieces)
        for label in t.labels:
            t.labels_pieces.append(label)
            if num_pieces > 1:
                if label[0] == "B":
                    _, tag = label.split(ner_prefix_joiner)
                    pad = f"I-{tag}"
                else:
                    pad = label
                t.labels_pieces += [pad] * (num_pieces - 1)


def enumerate_entities(example: Example):
    id2index = {}
    for i, entity in enumerate(sorted(example.entities, key=lambda e: e.tokens[0].index_rel)):
        id2index[entity.id] = i
        entity.index = i

    for arc in example.arcs:
        arc.head_index = id2index[arc.head]
        arc.dep_index = id2index[arc.dep]


def fit_encodings(
        examples: List[Example],
        min_label_freq: int = 3,
        label_other: str = "O",
) -> Tuple[Dict[str, int], Dict[str, int]]:
    ner_labels = defaultdict(int)
    re_labels = defaultdict(int)

    for x in examples:
        for t in x.tokens:
            for label in t.labels_pieces:
                if label != label_other:
                    ner_labels[label] += 1
        for arc in x.arcs:
            re_labels[arc.rel] += 1

    def build_enc(labels):
        enc = {label_other: 0}
        index = 1
        for l, count in labels.items():
            if count >= min_label_freq:
                enc[l] = index
                index += 1
        return enc

    ner_enc = build_enc(ner_labels)
    re_enc = build_enc(re_labels)
    return ner_enc, re_enc


def apply_encodings(examples: List[Example], ner_enc: Dict[str, int], re_enc: Dict[str, int]):
    unk_ner_labels = defaultdict(int)
    unk_re_labels = defaultdict(int)

    for x in examples:
        for t in x.tokens:
            t.label_ids = []
            for label in t.labels_pieces:
                if label in ner_enc.keys():
                    t.label_ids.append(ner_enc[label])
                else:
                    t.label_ids.append(0)
                    unk_ner_labels[label] += 1
        for arc in x.arcs:
            if arc.rel in re_enc.keys():
                arc.rel_id = re_enc[arc.rel]
            else:
                arc.rel_id = 0
                unk_re_labels[arc.rel] += 1

    print("unk_ner_labels:", unk_ner_labels)
    print("unk_re_labels:", unk_re_labels)


# def change_tokens_and_entities(x: Example) -> Example:
#     """
#     tokens = [иван иванов живёт в деревне жопа]
#     labels = [B_PER I_PER O O O B_LOC]
#     entities = [
#         Entity(tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=2),
#         Entity(tokens=[жопа], labels=[B_LOC], start_token_id=5, end_token_id=5),
#     ]
#
#
#     tokens = [PER живёт в деревне LOC]
#     labels = [B_PER I_PER O O O B_LOC]
#     entities = [
#         Entity(tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=0),
#         Entity(tokens=[жопа], labels=[B_LOC], start_token_id=4, end_token_id=4),
#     ]
#     """
#     x_new = deepcopy(x)
#     if x_new.entities:
#         entities_sorted = sorted(x_new.entities, key=lambda e: e.start_token_id)
#         pointers = [0]
#         tokens_new = []
#         for i, entity in enumerate(entities_sorted, 1):
#             end_prev = pointers[i - 1]
#             start_curr = entity.start_token_id
#             tokens_new += x.tokens[end_prev:start_curr]
#             label = '[{}]'.format(entity.labels[0].split('-')[1])
#             tokens_new.append(label)
#             start_new = end_new = len(tokens_new) - 1
#             end_curr = entity.end_token_id
#             pointers.append(end_curr + 1)
#             if i == len(entities_sorted):
#                 start = entity.end_token_id + 1
#                 end = len(x.tokens)
#                 tokens_new += x.tokens[start:end]
#             entity.start_token_id = start_new
#             entity.end_token_id = end_new
#         x_new.tokens = tokens_new
#     return x_new
#
#
# def convert_example_for_bert(x: Example, tokenizer, tag2token: dict, mode: str, no_rel_id: int = 0) -> List[Example]:
#     """
#     https://github.com/facebookresearch/SpanBERT/blob/10641ea3795771dd96e9e3e9ef0ead4f4f6a29d2/code/run_tacred.py#L116
#
#     tokens = [иван иванов живёт в деревне жопа]
#     labels = [B_PER I_PER O O O B_LOC]
#     entities = [
#         Entity(id=T1, tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=2),
#         Entity(id=T2, tokens=[жопа], labels=[B_LOC], start_token_id=5, end_token_id=5),
#     ]
#     arc = [ARC(id=R1, head=T1, dep=T2, rel=3)]
#
#     # [CLS] <контекст> [START_{HEAD или DEP}] <токены HEAD или DEP> [END_{HEAD или DEP}]
#     # <контекст> [START_{DEP или HEAD}] <токены DEP или HEAD> [END_{DEP или HEAD}] <контекст>
#     # [SEP] [таг левого операнда отношения (head)] [SEP] [таг правого операнда отношения (dep)] [SEP]
#     [
#         Example(
#             tokens = [
#                 [CLS] [START_HEAD] иван иванов [END_HEAD] живёт в деревне
#                 [START_DEP] жопа [END_DEP] [SEP] [HEAD-PER] [SEP] [DEP-LOC] [SEP]
#             ],
#             label = 3
#         ),
#         Example(
#             tokens = [
#                 [CLS] [START_DEP] иван иванов [END_DEP] живёт в деревне
#                 [START_HEAD] жопа [END_HEAD] [SEP] [HEAD-LOC] [SEP] [DEP-PER] [SEP]
#             ],
#             label = 0
#         )
#     ]
#
#     в tag2token должны быть токены берта для следующих спец. токенов:
#     1) '[START_HEAD]', '[END_HEAD]', '[START_DEP]', '[END_DEP]'
#     2) таги именных сущностей
#     """
#     assert mode in {BertEncodings.TEXT, BertEncodings.NER, BertEncodings.TEXT_NER, BertEncodings.NER_TEXT}
#
#     arc2rel = {}
#     for arc in x.arcs:
#         arc2rel[(arc.head, arc.dep)] = arc.rel
#
#     examples_new = []
#
#     START_HEAD_TOKEN = tag2token[SpecialSymbols.START_HEAD]
#     END_HEAD_TOKEN = tag2token[SpecialSymbols.END_HEAD]
#     START_DEP_TOKEN = tag2token[SpecialSymbols.START_DEP]
#     END_DEP_TOKEN = tag2token[SpecialSymbols.END_DEP]
#
#     token2pieces = {}
#
#     def get_pieces(token):
#         if token not in token2pieces:
#             token2pieces[token] = tokenizer.tokenize(token)
#         return token2pieces[token]
#
#     id_new = 0
#     num_entities = len(x.entities)
#     for id_head in range(num_entities):
#         for id_dep in range(num_entities):
#             if id_head == id_dep:
#                 continue
#             head = x.entities[id_head]
#             dep = x.entities[id_dep]
#
#             tag_head = "HEAD_" + head.labels[0].split('-')[1]
#             tag_dep = "DEP_" + dep.labels[0].split('-')[1]
#
#             TAG_HEAD_TOKEN = tag2token[tag_head]
#             TAG_DEP_TOKEN = tag2token[tag_dep]
#
#             tokens_new = [SpecialSymbols.CLS]
#
#             if mode in {BertEncodings.TEXT, BertEncodings.TEXT_NER}:
#                 # [HEAD_START] иван иванов [HEAD_END] живёт в деревне [DEP_START] жопа [DEP_END] [SEP]
#
#                 for i, pieces in enumerate(map(get_pieces, x.tokens)):
#
#                     if i == head.start_token_id:
#                         tokens_new.append(START_HEAD_TOKEN)
#                     if i == dep.start_token_id:
#                         tokens_new.append(START_DEP_TOKEN)
#
#                     tokens_new += pieces
#
#                     if i == head.end_token_id:
#                         tokens_new.append(END_HEAD_TOKEN)
#                     if i == dep.end_token_id:
#                         tokens_new.append(END_DEP_TOKEN)
#
#                 tokens_new.append(SpecialSymbols.SEP)
#
#                 if mode == BertEncodings.TEXT_NER:
#                     # + [HEAD_PER] [SEP] [DEP_LOC] [SEP]
#                     tokens_new += [TAG_HEAD_TOKEN, SpecialSymbols.SEP, TAG_DEP_TOKEN, SpecialSymbols.SEP]
#
#             else:
#                 # [HEAD_PER] живёт в деревне [DEP_LOC] [SEP]
#                 head_pieces = []
#                 dep_pieces = []
#
#                 for i, pieces in enumerate(map(get_pieces, x.tokens)):
#
#                     if i == head.start_token_id:
#                         tokens_new.append(TAG_HEAD_TOKEN)
#                     if i == dep.start_token_id:
#                         tokens_new.append(TAG_DEP_TOKEN)
#
#                     if head.start_token_id <= i <= head.end_token_id:
#                         head_pieces += pieces
#                     elif dep.start_token_id <= i <= dep.end_token_id:
#                         dep_pieces += pieces
#                     else:
#                         tokens_new += pieces
#
#                 tokens_new.append(SpecialSymbols.SEP)
#
#                 if mode == BertEncodings.NER_TEXT:
#                     # + [иван иванов [SEP] жопа [SEP]]
#                     tokens_new += head_pieces
#                     tokens_new.append(SpecialSymbols.SEP)
#                     tokens_new += dep_pieces
#                     tokens_new.append(SpecialSymbols.SEP)
#
#             token_ids = tokenizer.convert_tokens_to_ids(tokens_new)
#             rel = arc2rel.get((head.id, dep.id), no_rel_id)
#             # x.id = <название файла>_<порядковый номер предложения>
#             # id_new = <x.id>_<порядковый номер отношения>
#             x_new = Example(id=f'{x.id}_{id_new}', tokens=token_ids, label=rel)
#             examples_new.append(x_new)
#             id_new += 1
#     return examples_new
