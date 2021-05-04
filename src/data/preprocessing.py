from copy import deepcopy
from typing import List, Pattern, Tuple, Dict
from itertools import accumulate
from rusenttokenize import ru_sent_tokenize
from collections import defaultdict
import nltk

from src.data.base import (
    Entity,
    Example,
    Languages,
    NerEncodings,
    NerPrefixJoiners,
    TOKENS_EXPRESSION,
    Token,
    Span,
)
from src.utils import get_connected_components

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
        tokens_expression: Pattern = None,
        fix_pointers: bool = True
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
    :param fix_pointers нужно ли фиксить разбиения на предложения с учётом того, что граница не может проходить
     через именную сущность.

    в таком случае граница куска передвигается, что влечёт кейсы, где размер куска больше window.
    что не есть хорошо при инференсе на уровне документа.
    по этой причине было решено сначала фиксить pointers, а потом выводить куски над пофикшенными предложениями:
    см. fix_pointers_fn и get_sentences_spans_fixed_pointers
    :return:
    """
    # чтоб избавиться от warning "Expected type Optional[str], got (o: object)"
    # при создании итоговых инстансов класса Example
    assert isinstance(example.id, str)
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

    if fix_pointers:
        pointers = fix_pointers_fn(pointers=pointers, entity_spans=entity_spans)

    # TODO: делать это вне этой функции
    assign_sent_ids_to_tokens(example=example, pointers=pointers)

    if fix_pointers:
        sent_spans = get_sentences_spans(
            entity_spans=entity_spans,
            pointers=pointers,
            window=window,
            stride=stride
        )
    else:
        num_sentences = len(pointers) - 1
        sent_spans = get_sentences_spans_fixed_pointers(
            num_sentences=num_sentences,
            window=window,
            stride=stride
        )

    # print("spans_sents:", spans_sents)
    # print("spans_tokens:", spans_tokens)

    res = []
    for span in sent_spans:
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
                pieces=t.pieces.copy(),
                token_ids=t.token_ids.copy(),
                id_sent=t.id_sent
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
                    comment=entity.comment,
                    id_chain=entity.id_chain
                )
                entities.append(entity_new)
                entity_ids.add(entity.id)

        # events  TODO: сделать без deepcopy
        events = [deepcopy(event) for event in example.events if event.trigger in entity_ids]

        # arcs  TODO: сделать без deepcopy
        arcs = [deepcopy(arc) for arc in example.arcs if (arc.head in entity_ids) and (arc.dep in entity_ids)]

        id_child = f"{example.id}_{span.start}-{span.end}"
        example_copy = Example(
            filename=example.filename,
            id=id_child,
            text=text,
            tokens=tokens,
            entities=entities,
            events=events,
            arcs=arcs,
            label=example.label,
            parent=example.id,
        )

        res.append(example_copy)

    return res


def fix_pointers_fn(pointers: List[int], entity_spans: List[Span]) -> List[int]:
    res = []
    for p in pointers:
        is_good = True
        for span in entity_spans:
            if span.start < p <= span.end:
                is_good = False
                break
        if is_good:
            res.append(p)
    return res


def get_sentences_spans(entity_spans: List[Span], pointers: List[int], window: int = 1, stride: int = None) -> List[Span]:
    """
    предполагается, что pointers не пофикшены: то есть граница предложения может проходить через сущность.
    TODO: дописать описание

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
    bad_starts = set()  # индексы предложений, которые содержат только часть сущности
    while True:
        # print(span_ptr)
        end_token_id = pointers[end]

        for span in entity_spans:
            if span[0] < end_token_id <= span[1]:
                is_good_split = False
                break

        if is_good_split:
            res.append(Span(start=start, end=end))
            start = min(num_sentences - 1, start + stride)
            while start in bad_starts:
                start += 1
            end = min(num_sentences, start + window)
        else:
            bad_starts.add(end)
            end = min(num_sentences, end + 1)

        if end == num_sentences:
            res.append(Span(start=start, end=end))
            break

        # присвоение флагу is_good_split дефолтного значения
        is_good_split = True

    return res


def get_sentences_spans_fixed_pointers(num_sentences: int, window: int = 1, stride: int = None) -> List[Span]:
    # stride
    if stride is None:
        stride = window
    else:
        assert stride <= window

    if window >= num_sentences:
        return [Span(start=0, end=num_sentences)]

    res = []
    start = 0
    while True:
        end = start + window
        if end <= num_sentences:
            res.append(Span(start=start, end=end))
            start += stride
        else:
            break
    return res


def assign_sent_ids_to_tokens(example: Example, pointers: List[int]):
    num_sentences = len(pointers) - 1
    for i in range(num_sentences):
        start = pointers[i]
        end = pointers[i + 1]
        for t in example.tokens[start:end]:
            t.id_sent = i


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
            # (Иван, B-PER) -> ([Ив, #ан], [B-PER, I-PER])
            t.labels_pieces.append(label)
            if num_pieces > 1:
                if label[0] == "B":
                    _, tag = label.split(ner_prefix_joiner)
                    pad = f"I{ner_prefix_joiner}{tag}"
                else:
                    pad = label
                t.labels_pieces += [pad] * (num_pieces - 1)


def enumerate_entities(example: Example):
    id2index = {}
    entities_sorted = sorted(example.entities, key=lambda e: (e.tokens[0].index_rel, e.tokens[-1].index_rel))
    for i, entity in enumerate(entities_sorted):
        id2index[entity.id] = i
        entity.index = i

    for arc in example.arcs:
        arc.head_index = id2index[arc.head]
        arc.dep_index = id2index[arc.dep]


def fit_encodings(
        examples: List[Example],
        min_label_freq: int = 3,
        label_other: str = "O",
        ner_enc_token_level: bool = True
) -> Tuple[Dict[str, int], Dict[str, int]]:
    ner_labels = defaultdict(int)
    re_labels = defaultdict(int)

    for x in examples:
        if ner_enc_token_level:
            for t in x.tokens:
                for label in t.labels_pieces:
                    if label != label_other:
                        ner_labels[label] += 1
        else:
            for entity in x.entities:
                ner_labels[entity.label] += 1
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


def apply_encodings(
        examples: List[Example],
        ner_enc: Dict[str, int],
        re_enc: Dict[str, int],
        ner_enc_token_level: bool = True
):
    unk_ner_labels = defaultdict(int)
    unk_re_labels = defaultdict(int)

    for x in examples:
        if ner_enc_token_level:
            for t in x.tokens:
                t.label_ids = []
                for label in t.labels_pieces:
                    if label in ner_enc.keys():
                        t.label_ids.append(ner_enc[label])
                    else:
                        t.label_ids.append(0)
                        unk_ner_labels[label] += 1
        else:
            for entity in x.entities:
                if entity.label in ner_enc.keys():
                    entity.label_id = ner_enc[entity.label]
                else:
                    entity.label_id = 0
                    unk_ner_labels[entity.label] += 1
        for arc in x.arcs:
            if arc.rel in re_enc.keys():
                arc.rel_id = re_enc[arc.rel]
            else:
                arc.rel_id = 0
                unk_re_labels[arc.rel] += 1

    print("unk_ner_labels:", unk_ner_labels)
    print("unk_re_labels:", unk_re_labels)


def show_diff(reference: List[Example], proposed: List[Example]):
    r = sum(len(x.entities) for x in reference)
    p = sum(len(x.entities) for x in proposed)
    print("num entities reference:", r)
    print("num entities proposed:", p)
    print("percent change:", round((p / r - 1.0) * 100, 4), "%")

    r = sum(len(x.arcs) for x in reference)
    p = sum(len(x.arcs) for x in proposed)
    print("num relations reference:", r)
    print("num relations proposed:", p)
    print("percent change:", round((p / r - 1.0) * 100, 4), "%")


def assign_id_chain(examples: List[Example]):
    for x in examples:
        id2entity = {}
        g = {}
        for entity in x.entities:
            g[entity.id] = set()
            id2entity[entity.id] = entity
        for arc in x.arcs:
            g[arc.head].add(arc.dep)

        components = get_connected_components(g)

        for id_chain, comp in enumerate(components):
            for id_entity in comp:
                id2entity[id_entity].id_chain = id_chain
