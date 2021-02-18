import os
import re
import tqdm
import json
import shutil
import time
from copy import deepcopy
from typing import List, Dict, Tuple, Union
from itertools import accumulate
from collections import namedtuple, Counter, defaultdict
from rusenttokenize import ru_sent_tokenize
import nltk


# constants

TOKENS_EXPRESSION = re.compile(r"\w+|[^\w\s]")


class SpecialSymbols:
    CLS = '[CLS]'
    SEP = '[SEP]'
    START_HEAD = '[START_HEAD]'
    END_HEAD = '[END_HEAD]'
    START_DEP = '[START_DEP]'
    END_DEP = '[END_DEP]'


class BertEncodings:
    TEXT = "text"
    NER = "ner"
    TEXT_NER = "text_ner"
    NER_TEXT = "ner_text"


class NerEncodings:
    BIO = "bio"
    BILOU = "bilou"


class NerPrefixJoiners:
    UNDERSCORE = "_"  # WARNING: этот символ может встречаться в названиях сущностей/событий/отношений
    HYPHEN = "-"


class LineTypes:
    ENTITY = "T"
    EVENT = "E"
    RELATION = "R"
    ATTRIBUTE = "A"
    # https://brat.nlplab.org/standoff.html
    # For backward compatibility with existing standoff formats,
    # brat also recognizes the ID prefix "M" for attributes.
    ATTRIBUTE_OLD = "M"
    COMMENT = "#"
    EQUIV = "*"  # TODO: что это??


class Languages:
    EN = "en",
    RU = "ru"


# exceptions

class BadLineError(Exception):
    """
    строка файла .ann имеет неверный формат
    """


class EntitySpanError(Exception):
    """
    спану из файла .ann соответствует другая подстрока в файле .txt
    """


class NestedNerError(Exception):
    """
    одному токену соответствуют несколько лейблов
    """


class NestedNerSingleEntityTypeError(Exception):
    """
    одному токену соответствуют несколько лейблов сущности одного типа
    """


class RegexError(Exception):
    """регуляркой не получается токенизировать сущность: то есть expression.findall(entity_pattern) == []"""


# immutable structs

Attribute = namedtuple("Attribute", ["id", "type", "value"])
EventArgument = namedtuple("EventArgument", ["id", "role"])
Span = namedtuple("Span", ["start", "end"])


# mutable structs


class ReprMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        params_str = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f'{class_name}({params_str})'


class Token(ReprMixin):
    def __init__(
            self,
            text: str = None,
            span_abs: Span = None,
            span_rel: Span = None,
            index_abs: int = None,
            index_rel: int = None,
            labels: List[str] = None
    ):
        """

        :param text: текст
        :param span_abs: абсолютный* спан
        :param span_rel: относительный** спан
        :param index_abs: абсолютный* порядковый номер
        :param index_rel: относительный** порядковый номер
        :param labels: лейблы

        * на уровне документа
        ** на уровне примера
        """
        self.text = text
        self.span_abs = span_abs
        self.span_rel = span_rel
        self.index_abs = index_abs
        self.index_rel = index_rel  # пока не нужно
        self.labels = labels


class Entity(ReprMixin):
    def __init__(
            self,
            id: Union[int, str] = None,
            label: Union[int, str] = None,
            text: str = None,
            tokens: List[Token] = None,
            labels: List[str] = None,
            is_event_trigger: bool = False,
            attrs: List[Attribute] = None,  # атрибуты сущности
            comment: str = None
    ):
        self.id = id
        self.label = label
        self.text = text
        self.labels = labels
        self.tokens = tokens
        self.is_event_trigger = is_event_trigger
        self.attrs = attrs if attrs is not None else []
        self.comment = comment


class Event(ReprMixin):
    def __init__(
            self,
            id=None,
            trigger: str = None,
            label: str = None,
            args: List[EventArgument] = None,
            attrs: List[Attribute] = None,
            comment: str = None
    ):
        self.id = id
        self.trigger = trigger
        self.label = label
        self.args = args if args is not None else []
        self.attrs = attrs if attrs is not None else []
        self.comment = comment


class Arc(ReprMixin):
    def __init__(self, id, head, dep, rel, comment: str = None):
        self.id = id
        self.head = head
        self.dep = dep
        self.rel = rel
        self.comment = comment


class Example(ReprMixin):
    def __init__(
            self,
            filename: str = None,
            id: str = None,
            text: str = None,
            tokens: List[Token] = None,
            entities: List[Entity] = None,
            arcs: List[Arc] = None,
            events: List[Event] = None,  # пока только для дебага
            label: int = None,
    ):
        self.filename = filename
        self.id = id
        self.text = text
        self.tokens = tokens
        self.entities = entities
        self.arcs = arcs
        self.events = events if events is not None else []
        self.label = label  # в случае классификации предложений


# split


def split_example(example: Example, window=1, lang: str = Languages.RU) -> List[Example]:
    """
    Кусок исходного примера размером window предложений
    :param example: пример на уровне документа
    :param window: ширина окна на уровне предложений
    :param lang: язык
    :return:
    """
    if not example.text:
        print(f"[{example.id} WARNING]: empty text")
        return [Example(**example.__dict__)]

    if lang == Languages.RU:
        split_fn = ru_sent_tokenize
    else:
        split_fn = nltk.sent_tokenize

    sent_candidates = [sent for sent in split_fn(example.text) if len(sent) > 0]
    lengths = [len(TOKENS_EXPRESSION.findall(sent)) for sent in sent_candidates]
    assert sum(lengths) == len(example.tokens)

    spans_sents, spans_tokens = get_spans(example, lengths, window=window)

    res = []
    for i, ((start_sent, end_sent), (start_token, end_token)) in enumerate(zip(spans_sents, spans_tokens)):

        text = ' '.join(sent_candidates[start_sent:end_sent])
        # deepcopy - медленная штука
        example_copy = deepcopy(Example(
            filename=example.filename,
            id=example.id + "_" + str(i),
            text=text,
            tokens=example.tokens[start_token:end_token],
            entities=example.entities,
            events=example.events,
            arcs=example.arcs,
            label=example.label
        ))

        # tokens
        # TODO: рассмотреть случай, при котором text начинается с пробелов
        offset = example_copy.tokens[0].span_abs[0]
        for j, t in enumerate(example_copy.tokens):
            t.span_rel = t.span_abs[0] - offset, t.span_abs[1] - offset
            t.index_rel = j

        # entities
        entity_ids = set()
        entities = []
        for entity in example_copy.entities:
            if start_token <= entity.tokens[0].index_abs <= entity.tokens[-1].index_abs < end_token:
                entities.append(entity)
                entity_ids.add(entity.id)
        example_copy.entities = entities

        # events
        example_copy.events = [event for event in example_copy.events if event.trigger in entity_ids]

        # arcs
        example_copy.arcs = [arc for arc in example_copy.arcs if (arc.head in entity_ids) and (arc.dep in entity_ids)]

        res.append(example_copy)

    return res


def split_example_fast(example: Example, window=1, lang: str = Languages.RU) -> List[Example]:
    """
    Кусок исходного примера размером window предложений
    :param example: пример на уровне документа
    :param window: ширина окна на уровне предложений
    :param lang: язык
    :return:
    """
    if not example.text:
        print(f"[{example.id} WARNING]: empty text")
        return [Example(**example.__dict__)]

    if lang == Languages.RU:
        split_fn = ru_sent_tokenize
    else:
        split_fn = nltk.sent_tokenize

    sent_candidates = [sent for sent in split_fn(example.text) if len(sent) > 0]
    lengths = [len(TOKENS_EXPRESSION.findall(sent)) for sent in sent_candidates]
    assert sum(lengths) == len(example.tokens)

    spans_sents, spans_tokens = get_spans(example, lengths, window=window)

    # print("spans_sents:", spans_sents)
    # print("spans_tokens:", spans_tokens)

    res = []
    for i, ((start_sent, end_sent), (start_token, end_token)) in enumerate(zip(spans_sents, spans_tokens)):
        text = ' '.join(sent_candidates[start_sent:end_sent])

        # tokens
        # TODO: рассмотреть случай, при котором text начинается с пробелов
        # tokens = example.tokens[span.span_tokens[0]:span.span_tokens[1]]
        tokens = []
        # print(len(example.tokens), start_token)
        offset = example.tokens[start_token].span_abs[0]
        for j, t in enumerate(example.tokens[start_token:end_token]):
            t_copy = Token(
                text=t.text,
                span_abs=t.span_abs,
                span_rel=(t.span_abs[0] - offset, t.span_abs[1] - offset),
                index_abs=t.index_abs,
                index_rel=j,
                labels=t.labels.copy()
            )
            tokens.append(t_copy)

        # entities
        entity_ids = set()
        entities = []
        for entity in example.entities:
            if start_token <= entity.tokens[0].index_abs <= entity.tokens[-1].index_abs < end_token:
                entity_new = Entity(
                    id=entity.id,
                    label=entity.label,
                    text=entity.text,
                    tokens=tokens[start_token:end_token],
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
            id=example.id + "_" + str(i),
            text=text,
            tokens=tokens,
            entities=entities,
            events=events,
            arcs=arcs,
            label=example.label
        )

        res.append(example_copy)

    return res


# TODO: протестировать!
def get_spans(example, lengths, window=1) -> Tuple[List, List]:
    # entity_ptr = 0
    sent_starts = [0] + list(accumulate(lengths))
    spans_sents = []
    spans_tokens = []
    start = 0  # начало накопленной последовательности
    start_chunk = 0
    for i in range(len(lengths) - window + 1):
        # разделение на предложения плохое, если оно проходит через именную сущность
        is_bad_split = False
        start_next = sent_starts[i + window]
        start_next_chunk = i + window
        # TODO: можно оптимизировать, бегая каждый раз не по всем сущностям
        for entity in example.entities:
            # это условие имеет есто, если сущности сортированы по спану:
            # if entity.start_token_id >= start_next:
            #     break
            # начало находится в куске i, конец - в куске i + 1
            if entity.tokens[0].index_abs < start_next <= entity.tokens[-1].index_abs:
                is_bad_split = True
                break
        if not is_bad_split:
            spans_sents.append((start_chunk, start_next_chunk))
            spans_tokens.append((start, start_next))
            start = start_next
            start_chunk = i + 1
    return spans_sents, spans_tokens


# io


# TODO: сделать возможным создание одной последовательности ner-лейблов для именных сущностей и триггеров событий.
#  сейчас ner-лейблы событий всегда отделяются, хотя это оправдано только для вложенного ner-а.
# TODO: рассмотреть случай вложенного ner-a: сделать флаг allow_nested_ner. если True, то склеивать лейблы
#  токенов. если False, то вызывать ошибку при наличии пересекающихся спанов.
# TODO: в случае английского языка разделять тексты на предложения с помощью nltk


def load_examples(
        data_dir: str,
        n: int = None,
        split: bool = True,
        window: int = 1,
        fix_new_line_symbol: bool = True,
        ner_encoding: str = NerEncodings.BIO,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        allow_nested_ner: bool = False,
        allow_nested_similar_entities: bool = False,
        lang: str = Languages.RU
) -> List[Example]:
    """
    Загрузка примеров из brat-формата: пар (filename.txt, filename.ann)

    :param data_dir:
    :param n:
    :param split:
    :param window:
    :param fix_new_line_symbol:
    :param ner_encoding:
    :param ner_prefix_joiner:
    :param allow_nested_ner: может ли один токен относиться к нескольким сущностям
    :param allow_nested_similar_entities: может ли один токен относиться к нескольким сущностям одного типа
    :param lang:
    :return:
    """
    examples_doc = parse_collection(
        data_dir=data_dir,
        n=n,
        fix_new_line_symbol=fix_new_line_symbol,
        ner_encoding=ner_encoding,
        ner_prefix_joiner=ner_prefix_joiner,
        allow_nested_ner=allow_nested_ner,
        allow_nested_similar_entities=allow_nested_similar_entities
    )
    examples = []
    num_bad = 0
    num_examples = 0

    for x_raw in examples_doc:
        # проверяем целый пример
        try:
            check_example(example=x_raw)
        except AssertionError as e:
            print("[doc]", e)
            num_bad += 1
            continue

        if split:
            for x_raw_chunk in split_example(x_raw, window=window, lang=lang):
                num_examples += 1
                # проверяем кусок примера
                try:
                    check_example(example=x_raw_chunk)
                    examples.append(x_raw_chunk)
                except AssertionError as e:
                    print("[sent]", e)
                    num_bad += 1
        else:
            num_examples += 1
            examples.append(x_raw)
    print(f"{num_bad} / {num_examples} examples are bad")
    return examples


def save_examples(
        examples: List[Example],
        output_dir: str,
        id2relation: Dict[int, str],
        copy_texts: bool = False,
        collection_dir: str = None
):
    event_counter = defaultdict(int)
    filenames = set()
    for x in examples:
        filenames.add(x.filename)
        with open(os.path.join(output_dir, f"{x.filename}.ann"), "a") as f:
            events = {}
            # исходные сущности
            for entity in x.entities:
                line = f"{entity.id}\t{entity.label} {entity.start_index} {entity.end_index}\t{entity.text}\n"
                f.write(line)
                if entity.is_event_trigger:
                    if entity.id not in events:
                        id_event = event_counter[x.filename]
                        events[entity.id] = Event(
                            id=id_event,
                            trigger=entity.id,
                            label=entity.label,
                        )
                        event_counter[x.filename] += 1

            # отношения
            for arc in x.arcs:
                rel = id2relation[arc.rel]
                if arc.head in events:
                    arg = EventArgument(id=arc.dep, role=rel)
                    events[arc.head].args.append(arg)
                else:
                    id_arc = arc.id if isinstance(arc.id, str) else "R" + str(arc.id)
                    line = f"{id_arc}\t{rel} Arg1:{arc.head} Arg2:{arc.dep}\n"
                    f.write(line)

            # события
            for event in events.values():
                line = f"E{event.id}\t{event.label}:{event.trigger}"
                role2count = defaultdict(int)
                args_str = ""
                for arg in event.args:
                    i = role2count[arg.role]
                    role = arg.role
                    if i > 0:
                        role += str(i + 1)
                    args_str += f"{role}:{arg.id}" + ' '
                    role2count[arg.role] += 1
                args_str = args_str.strip()
                if args_str:
                    line += ' ' + args_str
                line += '\n'
                f.write(line)

    if copy_texts:
        assert collection_dir is not None
        for name in filenames:
            shutil.copy(os.path.join(collection_dir, f"{name}.txt"), output_dir)


def parse_collection(
        data_dir: str,
        n: int = None,
        fix_new_line_symbol: bool = True,
        ner_encoding: str = NerEncodings.BIO,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        allow_nested_ner: bool = False,
        allow_nested_similar_entities: bool = False
) -> List[Example]:
    """
    n - сколько примеров распарсить
    """

    # выбираем файлы, для которых есть исходный текст и разметка
    files = os.listdir(data_dir)
    texts = {x.split('.')[0] for x in files if x.endswith('.txt')}
    answers = {x.split('.')[0] for x in files if x.endswith('.ann')}
    names_to_use = sorted(texts & answers)  # сортировка для детерминированности
    print(f"num .txt files: {len(texts)}")
    print(f"num .ann files: {len(answers)}")
    print(f"num annotated texts: {len(names_to_use)}")

    names_to_parse = names_to_use[:n]

    # парсим примеры для обучения
    examples = []
    for filename in tqdm.tqdm(names_to_parse):
        try:
            example = parse_example(
                data_dir=data_dir,
                filename=filename,
                fix_new_line_symbol=fix_new_line_symbol,
                ner_encoding=ner_encoding,
                ner_prefix_joiner=ner_prefix_joiner,
                allow_nested_ner=allow_nested_ner,
                allow_nested_similar_entities=allow_nested_similar_entities
            )
            examples.append(example)
        except BadLineError as e:
            print(f"[{filename}] {e}")
            raise e
        except EntitySpanError as e:
            print(f"[{filename}]: example ignored; error message:")
            print(e)
            print("=" * 50)
            # raise e
        except NestedNerError as e:
            print(f"[{filename}] {e}")
            if not allow_nested_ner:
                raise e
        except NestedNerSingleEntityTypeError as e:
            print(f"[{filename}] {e}")
            if not allow_nested_similar_entities:
                raise e
        except RegexError as e:
            print(f"[{filename}] {e}")
            raise e
        except Exception as e:
            print(f"[{filename}] {e}")
            raise e
    print(f"successfully parsed {len(examples)} examples from {len(names_to_parse)} files.")
    return examples


def parse_example(
        data_dir: str,
        filename: str,
        fix_new_line_symbol: bool = False,
        ner_encoding: str = NerEncodings.BIO,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        allow_nested_ner: bool = False,
        allow_nested_similar_entities: bool = False
) -> Example:
    """
    строчка файла filename.ann:

    * сущность: T5\tBIN 325 337\tФормирование\n
    * отношение: R105\tTSK Arg1:T370 Arg2:T371\n
    * событие: E0\tBankruptcy:T0 Bankrupt:T1 Bankrupt2:T2\n
    * атрибут: A1\tEvent-time E0 Past\n
    * комментарий: #0\tAnnotatorNotes T3\tfoobar\n

    замечения по файлам .ann:
    1. сначала пишется триггер события, а потом событие:
    T12     Bankruptcy 1866 1877    банкротства
    E3      Bankruptcy:T12

    2. каждому событию соответствует одна пара (T, E), даже если они имеют общий триггер:
    T12     Bankruptcy 1866 1877    банкротства
    E3      Bankruptcy:T12
    T13     AnotherEvent 1866 1877    банкротства
    E4      AnotherEvent:T13

    3.1. аргумент атрибута или комментария всегда указан раньше, чем сам атрибут.
    3.2. если комментируется события, то аргументом комментария является идентификатор события,
    а не идекнтификатор триггера.
    T12     Bankruptcy 1866 1877    банкротства
    E3      Bankruptcy:T12
    A10     Negation E3
    #1      AnnotatorNotes E3  foobar

    данные наблюдения позволяют за один проход по всем строчка файла .ann сделать всё необходимое
    """
    # подгрузка текста
    with open(os.path.join(data_dir, f'{filename}.txt')) as f:
        text = ' '.join(f)
        if fix_new_line_symbol:
            text = text.replace('\n ', '\n')

    # токенизация
    tokens = []
    span2token = {}
    no_labels = ["O"]

    # бывают странные ситуации:
    # @ подстрока текста: передачи данных___________________7;
    # @ в файле .ann есть сущность "данных"
    # @ TOKENS_EXPRESSION разбивает на токены так: [передачи, данных___________________7]
    # @ получается невозможно определить индекс токена "данных"
    # @ будем в таком случае пытаться это сделать по индексу начала
    # start2index = {}
    for i, m in enumerate(TOKENS_EXPRESSION.finditer(text)):
        span = m.span()
        token = Token(
            text=m.group(),
            span_abs=span,
            span_rel=span,
            index_abs=i,
            index_rel=i,
            labels=no_labels
        )
        tokens.append(token)
        span2token[span] = token

    # .ann
    id2entity = {}
    id2event = {}
    id2arc = {}
    id2arg = {}
    with open(os.path.join(data_dir, f'{filename}.ann'), 'r') as f:
        for line in f:
            line = line.strip()
            content = line.split('\t')
            line_tag = content[0]
            line_type = line_tag[0]

            # сущность
            if line_type == LineTypes.ENTITY:
                # проверка того, что формат строки верный
                try:
                    _, entity, expected_entity_pattern = content
                except ValueError:
                    raise BadLineError(f"[{filename}]: something is wrong with line: {line}")

                entity_label, start_index, end_index = entity.split()
                start_index = int(start_index)
                end_index = int(end_index)

                # проверка того, что в файле .txt в спане из файла .ann находится
                # правильная именная сущность
                actual_entity_pattern = text[start_index:end_index]
                if actual_entity_pattern != expected_entity_pattern:
                    raise EntitySpanError(f"[{filename}]: something is wrong with markup; "
                                              f"expected entity is {expected_entity_pattern}, "
                                              f"but got {actual_entity_pattern}")

                entity_matches = list(TOKENS_EXPRESSION.finditer(expected_entity_pattern))

                num_entity_tokens = len(entity_matches)
                if num_entity_tokens == 0:
                    raise RegexError(f"regex fail to tokenize entity pattern {expected_entity_pattern}")

                entity_labels = []
                entity_tokens = []
                for i, m in enumerate(entity_matches):
                    # добавление токена сущности
                    # token = m.group()
                    # entity_tokens.append(token)

                    # вывод префикса:
                    prefix = get_label_prefix(
                        entity_token_index=i,
                        num_entity_tokens=num_entity_tokens,
                        ner_encoding=ner_encoding
                    )

                    # добавление лейбла
                    label = prefix + ner_prefix_joiner + entity_label
                    entity_labels.append(label)

                    # вывод спана токена в исходном тексте
                    si, ei = m.span()
                    token_span_abs = start_index + si, start_index + ei

                    try:
                        # вывод порядкового номера токена
                        # выполненное условие actual_entity_pattern == text[start_index:end_index]
                        # гарантирует отсутствие KeyError здесь:
                        token = span2token[token_span_abs]
                    except KeyError:
                        s, e = token_span_abs
                        msg = "can not infer token id from span or span is a part of token\n"
                        msg += f"absolute span: {token_span_abs}\n"
                        msg += f'entity token: <bos>{token}<eos>\n'
                        msg += f'corresponding text token: <bos>{text[s:e]}<eos>\n'
                        msg += f'context: {text[max(0, s - 50):s]}<bos>{text[s:e]}<eos>{text[e:e + 50]}'
                        raise EntitySpanError(msg)

                    # запись лейблов
                    if token.labels == no_labels:
                        token.labels = [label]
                    elif allow_nested_ner:
                        token_entity_labels = {l.split(ner_prefix_joiner)[-1] for l in token.labels}
                        if entity_label not in token_entity_labels:
                            token.labels.append(label)
                        else:
                            if allow_nested_similar_entities:
                                token.labels.append(label)
                            else:
                                raise NestedNerSingleEntityTypeError(
                                    f"[{filename}] tried to assign more than one label "
                                    f"of entity {entity_label} to token {token}"
                                )
                    else:
                        raise NestedNerError(f"[{filename}] tried to assign more than one label to token {token}")

                    # добавление токена
                    entity_tokens.append(token)

                # assert TOKENS_EXPRESSION.findall(actual_entity_pattern) == entity_tokens

                # создание сущности
                entity = Entity(
                    id=line_tag,
                    label=entity_label,
                    text=actual_entity_pattern,
                    tokens=entity_tokens,
                    labels=entity_labels,
                    is_event_trigger=False  # заполнится потом
                )
                id2entity[entity.id] = entity
                id2arg[entity.id] = entity

            # отношение
            elif line_type == LineTypes.RELATION:
                try:
                    _, relation = content
                    re_label, arg1, arg2 = relation.split()
                except ValueError:
                    raise BadLineError(f"[{filename}]: something is wrong with line: {line}")
                arc = Arc(
                    id=line_tag,
                    head=arg1.split(":")[1],
                    dep=arg2.split(":")[1],
                    rel=re_label
                )
                id2arc[arc.id] = arc
                id2arg[arc.id] = arc

            # событие
            elif line_type == LineTypes.EVENT:
                # E0\tBankruptcy:T0 Bankrupt:T1 Bankrupt2:T2
                event_args = content[1].split()
                event_trigger = event_args.pop(0)
                event_name, id_head = event_trigger.split(":")
                event = Event(
                    id=line_tag,
                    trigger=id_head,
                    label=event_name,
                )
                for dep in event_args:
                    rel, id_dep = dep.split(":")

                    # если аргументов одной роли несколько, то всем, начиная со второго,
                    # приписывается в конце номер (см. пример)
                    rel = re.sub(r'\d+', '', rel)

                    # запись отношения
                    # id должен быть уникальным
                    id_arc = f"{line_tag}_{id_dep}"
                    arc = Arc(
                        id=id_arc,
                        head=id_head,
                        dep=id_dep,
                        rel=rel
                    )
                    id2arc[id_arc] = arc

                    # запись аргумента события
                    arg = EventArgument(id=id_dep, role=rel)
                    event.args.append(arg)

                id2event[event.id] = event
                id2arg[event.id] = event

            # атрибут
            elif line_type == LineTypes.ATTRIBUTE or line_type == LineTypes.ATTRIBUTE_OLD:
                # A1\tEvent-time E0 Past - multi-value
                # A1\tNegation E12  - binary
                params = content[1].split()
                if len(params) == 3:  # multi-value
                    attr_type, id_arg, value = params
                    attr = Attribute(id=line_tag, type=attr_type, value=value)
                elif len(params) == 2:  # binary
                    flag, id_arg = params
                    attr = Attribute(id=line_tag, type=flag, value=True)
                else:
                    raise BadLineError(f"strange attribute line: {line}")

                try:
                    id2arg[id_arg].attrs.append(attr)
                except KeyError:
                    raise BadLineError("there is no arg for provided attr")

            # комментарии.
            elif line_type == LineTypes.COMMENT:
                # #0\tAnnotatorNotes T3\tfoobar\n
                _, id_arg = content[1].split()
                msg = content[2]
                try:
                    id2arg[id_arg].comment = msg
                except KeyError:
                    raise BadLineError("there is no arg for provided comment")

            # TODO: разобраться с этим
            elif line_type == LineTypes.EQUIV:
                pass

            else:
                raise Exception(f"invalid line: {line}")

    # оказывается, событие может быть указано раньше триггера в файле .ann
    for event in id2event.values():
        id2entity[event.trigger].is_event_trigger = True

    entities = list(id2entity.values())
    events = list(id2event.values())
    arcs = list(id2arc.values())

    example = Example(
        filename=filename,
        id=filename,
        text=text,
        tokens=tokens,
        entities=entities,
        arcs=arcs,
        events=events
    )

    return example


def get_label_prefix(entity_token_index: int, num_entity_tokens: int, ner_encoding: str):
    if ner_encoding == NerEncodings.BIO:
        if entity_token_index == 0:
            prefix = "B"
        else:
            prefix = "I"
    elif ner_encoding == NerEncodings.BILOU:
        if num_entity_tokens == 1:
            prefix = "U"
        else:
            if entity_token_index == 0:
                prefix = "B"
            elif entity_token_index == num_entity_tokens - 1:
                prefix = "L"
            else:
                prefix = "I"
    else:
        raise
    return prefix


# check


def check_example(example: Example, ner_encoding: str = NerEncodings.BIO):
    pass


# # TODO: адаптировать
# def check_example(example: Example, ner_encoding: str = NerEncodings.BIO):
#     """
#     NER:
#     * число токенов равно числу лейблов
#     * entity.start >= entity.end
#     * начало сущности >= 0, конец сущности < len(tokens)
#     RE:
#     * оба аргумента отношений есть в entities
#     """
#     # обязателен айдишник
#     assert example.id is not None, f"example {example} has no id!"
#     prefix = f"[{example.id}]: "
#
#     # ner-кодировка
#     assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}, \
#         f"expected ner_encoding {NerEncodings.BIO} or {NerEncodings.BILOU}, got {ner_encoding}"
#
#     num_tokens = len(example.tokens)
#
#     # # биекция между токенами и лейблами
#     # assert num_tokens == len(example.labels), \
#     #     prefix + f"tokens and labels mismatch, {num_tokens} != {len(example.labels)}"
#
#     entity_ids_all = set()
#     entity_ids_wo_events = set()
#     entity_spans = set()
#     event2entities = defaultdict(set)
#     for entity in example.entities:
#         # обязателен айдишник
#         assert entity.id is not None, \
#             prefix + f"[{entity}] entity has no id!"
#
#         # проверка валидности спана
#         assert 0 <= entity.start_token_id <= entity.end_token_id < num_tokens, \
#             prefix + f"[{entity}] strange entity span: " \
#                 f"start token id: {entity.start_token_id}, end token id: {entity.end_token_id}. num tokens: {num_tokens}"
#
#         # проверка корректности соответстия токенов сущности токенам примера
#         expected_tokens = example.tokens[entity.start_token_id:entity.end_token_id + 1]
#         assert expected_tokens == entity.tokens, \
#             prefix + f"[{entity}] tokens and example tokens mismatch: {entity.tokens} != {expected_tokens}"
#
#         # # проверка корректности соответстия лейблов сущности лейблам примера
#         # if entity.is_event_trigger:
#         #     ner_labels = example.labels_events[entity.label]
#         # else:
#         #     ner_labels = example.labels
#         # expected_labels = ner_labels[entity.start_token_id:entity.end_token_id + 1]
#         # assert expected_labels == entity.labels, \
#         #     prefix + f"[{entity}]: labels and example labels mismatch: {entity.labels} != {expected_labels}"
#
#         # кэш
#         entity_ids_all.add(entity.id)
#         entity_spans.add((entity.start_token_id, entity.end_token_id))
#         if entity.is_event_trigger:
#             event2entities[entity.label].add(entity.id)
#         else:
#             entity_ids_wo_events.add(entity.id)
#
#     # проверка уникальности сущностей
#     assert len(example.entities) == len(entity_ids_all), \
#         prefix + f"entity ids are not unique: {len(example.entities)} != {len(entity_ids_all)}"
#
#     # проверка биекции между множеством спанов и множеством сущностей.
#     # пока предполагается её наличие.
#     # @ каждому событию соответствет своя сущность;
#     # @ каждой сущности соответствует свой спан;
#     # @ если одному паттерну соответствуют несколько событий, то ему могут соответствовать несколько спанов
#     # @ пока такое возможно
#     # assert len(example.entities) == len(entity_spans), \
#     #     prefix + f"there are span duplicates: " \
#     #     f"number of entities is {len(example.entities)}, but number of unique text spans is {len(entity_spans)}"
#
#     def check_ner_labels(ent_ids, labels, ner_label_other):
#         """проверка непротиворечивости множества сущностей лейблам"""
#         if len(ent_ids) == 0:
#             assert set(labels) == {ner_label_other}, \
#                 prefix + f"ner labels and named entities mismatch: ner labels are {set(labels)}, " \
#                 f"but there are no entities in example."
#         else:
#             assert set(labels) != {ner_label_other}, \
#                 prefix + f"ner labels and named entities mismatch: ner labels are {set(labels)}, " \
#                 f"but there are following entities in example: {ent_ids}"
#
#     # check_ner_labels(ent_ids=entity_ids_wo_events, labels=example.labels, ner_label_other=self.ner_label_other)
#     #
#     # for k, v in event2entities.items():
#     #     check_ner_labels(ent_ids=v, labels=example.labels_events[k], ner_label_other=self.ner_label_other)
#
#     arc_args = []
#     for arc in example.arcs:
#         # проверка того, что в примере есть исходящая вершина
#         assert arc.head in entity_ids_all, \
#             prefix + f"something is wrong with arc {arc.id}: head {arc.head} is unknown"
#         # проверка того, что в примере есть входящая вершина
#         assert arc.dep in entity_ids_all, \
#             prefix + f"something is wrong with arc {arc.id}: dep {arc.dep} is unknown"
#         arc_args.append((arc.head, arc.dep))
#     # проверка того, что одному ребру соответствует одно отношение
#     if len(arc_args) != len(set(arc_args)):
#         arc_counts = {k: v for k, v in Counter(arc_args).items() if v > 1}
#         raise AssertionError(prefix + f'there duplicates in arc args: {arc_counts}')
#
#     num_triggers = sum(entity.is_event_trigger for entity in example.entities)
#     num_events = len(example.events)
#     assert num_events == num_triggers, \
#         prefix + f"number of event triggers is {num_triggers}, but number of events is {num_events}"
#
#     if ner_encoding == NerEncodings.BILOU:
#         # проверка того, что число начал сущности равно числу концов
#         num_start_ids = sum(x.startswith("B") for x in example.labels)
#         num_end_ids = sum(x.startswith("L") for x in example.labels)
#         assert num_start_ids == num_end_ids, \
#             prefix + f"num start ids: {num_start_ids}, num end ids: {num_end_ids}"


def _check_ner(example: Example):
    """
    1. число токенов = число лейблов
    2. индекс начала сущности <= индекс конца сущности
    3. символ начала сущности <= символ конца сущности
    4. entity.text == example.text[entity.span_rel[0]:entity.stan_rel[1]]
    """
    for entity in example.entities:
        pass


# encoding


class Vocab(ReprMixin):
    def __init__(self, values):
        if isinstance(values, dict):  # str -> int
            self._value2id = values
            self._id2value = {v: k for k, v in values.items()}
        elif isinstance(values, set):
            special_value = "O"
            values -= {special_value}
            self._id2value = dict(enumerate(sorted(values), 1))
            self._id2value[0] = special_value
            self._value2id = {v: k for k, v in self._id2value.items()}
        else:
            raise

    @property
    def size(self):
        return len(self._id2value)

    @property
    def encodings(self):
        return self._value2id

    @property
    def inv_encodings(self):
        return self._id2value

    def get_value(self, id):
        return self._id2value[id]

    def get_id(self, value):
        return self._value2id[value]


# class ExampleEncoder:
#     def __init__(
#             self,
#             ner_encoding: str = NerEncodings.BIO,
#             ner_label_other: str = "O",
#             re_label_other: str = "O",
#             ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
#             add_seq_bounds: bool = True
#     ):
#         assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}
#         self.ner_encoding = ner_encoding
#         self.ner_label_other = ner_label_other
#         self.re_label_other = re_label_other
#         self.add_seq_bounds = add_seq_bounds
#         self.ner_prefix_joiner = ner_prefix_joiner
#
#         self.vocab_ner = None
#         self.vocab_re = None
#         self.vocabs_events = {}
#
#     def fit_transform(self, examples):
#         self.fit(examples)
#         return self.transform(examples)
#
#     def fit(self, examples):
#         # инициализация значений словаря
#         vocab_ner = set()
#         vocabs_events = defaultdict(set)
#
#         prefixes = {"B", "I"}
#         if self.ner_encoding == NerEncodings.BILOU:
#             prefixes |= {"L", "U"}
#
#         def extend_vocab(label_, ner_prefix_joiner, vocab_values):
#             if ner_prefix_joiner in label_:
#                 # предполагаем, что каждая сущность может состоять из нескольких токенов
#                 label_ = label_.split(ner_prefix_joiner)[-1]
#                 for p in prefixes:
#                     vocab_values.add(p + ner_prefix_joiner + label_)
#             else:
#                 vocab_values.add(label_)
#
#         for x in examples:
#             for label in x.labels:
#                 extend_vocab(label, self.ner_prefix_joiner, vocab_ner)
#             for event_tag, labels in x.labels_events.items():
#                 for label in labels:
#                     extend_vocab(label, self.ner_prefix_joiner, vocabs_events[event_tag])
#
#         vocab_ner.add(self.ner_label_other)
#         self.vocab_ner = Vocab(vocab_ner)
#
#         self.vocabs_events = {}
#         for k, v in vocabs_events.items():
#             v.add(self.ner_label_other)
#             self.vocabs_events[k] = Vocab(v)
#
#         # arcs vocab
#         vocab_re = set()
#         for x in examples:
#             for arc in x.arcs:
#                 vocab_re.add(arc.rel)
#         vocab_re.add(self.re_label_other)
#         self.vocab_re = Vocab(vocab_re)
#
#     def transform(self, examples: List[Example]) -> List[Example]:
#         res = []
#         for x in examples:
#             x_enc = self.transform_example(x)
#             res.append(x_enc)
#         return res
#
#     def transform_example(self, example: Example) -> Example:
#         """
#         Кодирование категориальных атрибутов примеров:
#         * tokens - List[str] (остаётся неизменным)
#         * labels - List[int]
#         * entities - List[Tuple[start, end]]
#         * arcs - List[Tuple[head, dep, id_relation]]
#         """
#         example_enc = Example(
#             filename=example.filename,
#             id=example.id,
#             text=example.text
#         )
#
#         # tokens
#         example_enc.tokens = example.tokens.copy()
#         if self.add_seq_bounds:
#             example_enc.tokens = ["[START]"] + example_enc.tokens + ["[END]"]
#
#         # tokens spans
#         example_enc.tokens_spans = example.tokens_spans.copy()
#         if self.add_seq_bounds:
#             example_enc.tokens_spans = [(-1, -1)] + example_enc.tokens_spans + [(-1, -1)]  # TODO: ок ли так делать?
#
#         # labels
#         def encode_labels(labels, vocab, add_seq_bounds, ner_label_other):
#             labels_encoded = []
#             for label in labels:
#                 label_enc = vocab.get_id(label)
#                 labels_encoded.append(label_enc)
#             if add_seq_bounds:
#                 label = vocab.get_id(ner_label_other)
#                 labels_encoded = [label] + labels_encoded + [label]
#             # example_enc.labels = labels_encoded
#             return labels_encoded
#
#         example_enc.labels = encode_labels(
#             labels=example.labels, vocab=self.vocab_ner,
#             add_seq_bounds=self.add_seq_bounds, ner_label_other=self.ner_label_other
#         )
#         example_enc.labels_events = {}
#         for k, v in example.labels_events.items():
#             example_enc.labels_events[k] = encode_labels(
#                 labels=v, vocab=self.vocabs_events[k],
#                 add_seq_bounds=self.add_seq_bounds, ner_label_other=self.ner_label_other
#             )
#
#         # entities
#         example_enc.entities = deepcopy(example.entities)
#         if self.add_seq_bounds:
#             # потому что в начало добавлен токен начала строки
#             for entity in example_enc.entities:
#                 entity.start_token_id += 1
#                 entity.end_token_id += 1
#
#         # arcs
#         arcs_encoded = []
#         for arc in example.arcs:
#             id_rel = self.vocab_re.get_id(arc.rel)
#             arc_enc = Arc(id=arc.id, head=arc.head, dep=arc.dep, rel=id_rel)
#             arcs_encoded.append(arc_enc)
#         example_enc.arcs = arcs_encoded
#         return example_enc
#
#     def save(self, encoder_dir):
#         d = {
#             "ner_encoding": self.ner_encoding,
#             "ner_label_other": self.ner_label_other,
#             "re_label_other": self.re_label_other,
#             "ner_prefix_joiner": self.ner_prefix_joiner,
#             "add_seq_bounds": self.add_seq_bounds
#         }
#         with open(os.path.join(encoder_dir, "encoder_config.json"), "w") as f:
#             json.dump(d, f, indent=4)
#
#         with open(os.path.join(encoder_dir, "ner_encodings.json"), "w") as f:
#             json.dump(self.vocab_ner.encodings, f, indent=4)
#
#         with open(os.path.join(encoder_dir, "ner_encodings_events.json"), "w") as f:
#             json.dump({k: v.encodings for k, v in self.vocabs_events.items()}, f, indent=4)
#
#         with open(os.path.join(encoder_dir, "re_encodings.json"), "w") as f:
#             json.dump(self.vocab_re.encodings, f, indent=4)
#
#     @classmethod
#     def load(cls, encoder_dir):
#         config = json.load(open(os.path.join(encoder_dir, "encoder_config.json")))
#         enc = cls(**config)
#
#         ner_encodings = json.load(open(os.path.join(encoder_dir, "ner_encodings.json")))
#         enc.vocab_ner = Vocab(values=ner_encodings)
#
#         re_encodings = json.load(open(os.path.join(encoder_dir, "re_encodings.json")))
#         enc.vocab_re = Vocab(values=re_encodings)
#
#         d = json.load(open(os.path.join(encoder_dir, "ner_encodings_events.json")))
#         enc.vocabs_events = {k: Vocab(values=v) for k, v in d.items()}
#
#         return enc


# преобразование примеров


def change_tokens_and_entities(x: Example) -> Example:
    """
    tokens = [иван иванов живёт в деревне жопа]
    labels = [B_PER I_PER O O O B_LOC]
    entities = [
        Entity(tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=2),
        Entity(tokens=[жопа], labels=[B_LOC], start_token_id=5, end_token_id=5),
    ]


    tokens = [PER живёт в деревне LOC]
    labels = [B_PER I_PER O O O B_LOC]
    entities = [
        Entity(tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=0),
        Entity(tokens=[жопа], labels=[B_LOC], start_token_id=4, end_token_id=4),
    ]
    """
    x_new = deepcopy(x)
    if x_new.entities:
        entities_sorted = sorted(x_new.entities, key=lambda e: e.start_token_id)
        pointers = [0]
        tokens_new = []
        for i, entity in enumerate(entities_sorted, 1):
            end_prev = pointers[i - 1]
            start_curr = entity.start_token_id
            tokens_new += x.tokens[end_prev:start_curr]
            label = '[{}]'.format(entity.labels[0].split('-')[1])
            tokens_new.append(label)
            start_new = end_new = len(tokens_new) - 1
            end_curr = entity.end_token_id
            pointers.append(end_curr + 1)
            if i == len(entities_sorted):
                start = entity.end_token_id + 1
                end = len(x.tokens)
                tokens_new += x.tokens[start:end]
            entity.start_token_id = start_new
            entity.end_token_id = end_new
        x_new.tokens = tokens_new
    return x_new


def convert_example_for_bert(x: Example, tokenizer, tag2token: dict, mode, no_rel_id=0) -> List[Example]:
    """
    https://github.com/facebookresearch/SpanBERT/blob/10641ea3795771dd96e9e3e9ef0ead4f4f6a29d2/code/run_tacred.py#L116

    tokens = [иван иванов живёт в деревне жопа]
    labels = [B_PER I_PER O O O B_LOC]
    entities = [
        Entity(id=T1, tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=2),
        Entity(id=T2, tokens=[жопа], labels=[B_LOC], start_token_id=5, end_token_id=5),
    ]
    arc = [ARC(id=R1, head=T1, dep=T2, rel=3)]

    # [CLS] <контекст> [START_{HEAD или DEP}] <токены HEAD или DEP> [END_{HEAD или DEP}]
    # <контекст> [START_{DEP или HEAD}] <токены DEP или HEAD> [END_{DEP или HEAD}] <контекст>
    # [SEP] [таг левого операнда отношения (head)] [SEP] [таг правого операнда отношения (dep)] [SEP]
    [
        Example(
            tokens = [
                [CLS] [START_HEAD] иван иванов [END_HEAD] живёт в деревне
                [START_DEP] жопа [END_DEP] [SEP] [HEAD-PER] [SEP] [DEP-LOC] [SEP]
            ],
            label = 3
        ),
        Example(
            tokens = [
                [CLS] [START_DEP] иван иванов [END_DEP] живёт в деревне
                [START_HEAD] жопа [END_HEAD] [SEP] [HEAD-LOC] [SEP] [DEP-PER] [SEP]
            ],
            label = 0
        )
    ]

    в tag2token должны быть токены берта для следующих спец. токенов:
    1) '[START_HEAD]', '[END_HEAD]', '[START_DEP]', '[END_DEP]'
    2) таги именных сущностей
    """
    assert mode in {BertEncodings.TEXT, BertEncodings.NER, BertEncodings.TEXT_NER, BertEncodings.NER_TEXT}

    arc2rel = {}
    for arc in x.arcs:
        arc2rel[(arc.head, arc.dep)] = arc.rel

    examples_new = []

    START_HEAD_TOKEN = tag2token[SpecialSymbols.START_HEAD]
    END_HEAD_TOKEN = tag2token[SpecialSymbols.END_HEAD]
    START_DEP_TOKEN = tag2token[SpecialSymbols.START_DEP]
    END_DEP_TOKEN = tag2token[SpecialSymbols.END_DEP]

    token2pieces = {}

    def get_pieces(token):
        if token not in token2pieces:
            token2pieces[token] = tokenizer.tokenize(token)
        return token2pieces[token]

    id_new = 0
    num_entities = len(x.entities)
    for id_head in range(num_entities):
        for id_dep in range(num_entities):
            if id_head == id_dep:
                continue
            head = x.entities[id_head]
            dep = x.entities[id_dep]

            tag_head = "HEAD_" + head.labels[0].split('-')[1]
            tag_dep = "DEP_" + dep.labels[0].split('-')[1]

            TAG_HEAD_TOKEN = tag2token[tag_head]
            TAG_DEP_TOKEN = tag2token[tag_dep]

            tokens_new = [SpecialSymbols.CLS]

            if mode in {BertEncodings.TEXT, BertEncodings.TEXT_NER}:
                # [HEAD_START] иван иванов [HEAD_END] живёт в деревне [DEP_START] жопа [DEP_END] [SEP]

                for i, pieces in enumerate(map(get_pieces, x.tokens)):

                    if i == head.start_token_id:
                        tokens_new.append(START_HEAD_TOKEN)
                    if i == dep.start_token_id:
                        tokens_new.append(START_DEP_TOKEN)

                    tokens_new += pieces

                    if i == head.end_token_id:
                        tokens_new.append(END_HEAD_TOKEN)
                    if i == dep.end_token_id:
                        tokens_new.append(END_DEP_TOKEN)

                tokens_new.append(SpecialSymbols.SEP)

                if mode == BertEncodings.TEXT_NER:
                    # + [HEAD_PER] [SEP] [DEP_LOC] [SEP]
                    tokens_new += [TAG_HEAD_TOKEN, SpecialSymbols.SEP, TAG_DEP_TOKEN, SpecialSymbols.SEP]

            else:
                # [HEAD_PER] живёт в деревне [DEP_LOC] [SEP]
                head_pieces = []
                dep_pieces = []

                for i, pieces in enumerate(map(get_pieces, x.tokens)):

                    if i == head.start_token_id:
                        tokens_new.append(TAG_HEAD_TOKEN)
                    if i == dep.start_token_id:
                        tokens_new.append(TAG_DEP_TOKEN)

                    if head.start_token_id <= i <= head.end_token_id:
                        head_pieces += pieces
                    elif dep.start_token_id <= i <= dep.end_token_id:
                        dep_pieces += pieces
                    else:
                        tokens_new += pieces

                tokens_new.append(SpecialSymbols.SEP)

                if mode == BertEncodings.NER_TEXT:
                    # + [иван иванов [SEP] жопа [SEP]]
                    tokens_new += head_pieces
                    tokens_new.append(SpecialSymbols.SEP)
                    tokens_new += dep_pieces
                    tokens_new.append(SpecialSymbols.SEP)

            token_ids = tokenizer.convert_tokens_to_ids(tokens_new)
            rel = arc2rel.get((head.id, dep.id), no_rel_id)
            # x.id = <название файла>_<порядковый номер предложения>
            # id_new = <x.id>_<порядковый номер отношения>
            x_new = Example(id=f'{x.id}_{id_new}', tokens=token_ids, label=rel)
            examples_new.append(x_new)
            id_new += 1
    return examples_new


def convert_tokens_to_pieces(x, tokenizer):
    x_new = deepcopy(x)

    labels_new = []
    pieces_list = []

    for token, label in zip(x.tokens, x.labels):
        pieces = tokenizer.tokenize(token)
        pieces_list.append(pieces)
        num_pieces = len(pieces)
        if label == 'O':
            labels_new += [label] * num_pieces
        else:
            tag = label.split('-')[-1]
            if label[0] == 'B':
                labels_new += ["B-" + tag] + ["I-" + tag] * (num_pieces - 1)
            elif label[0] == 'I':
                labels_new += ["I-" + tag] * num_pieces
            elif label[0] == 'L':
                labels_new += ["I-" + tag] * (num_pieces - 1) + ['L-' + tag]
            elif label[0] == 'U':
                if num_pieces == 1:
                    labels_new.append(label)
                else:
                    labels_new += ["B-" + tag] + ["I-" + tag] * (num_pieces - 2) + ['L-' + tag]

    x_new.labels = labels_new

    if x.entities:

        tokens_new = []
        pointers = [0]

        for entity in sorted(x_new.entities, key=lambda x: x.start_token_id):
            start_init = entity.start_token_id
            end_init = entity.end_token_id

            start = pointers[-1]
            end = start_init
            for pieces in pieces_list[start:end]:
                tokens_new += pieces
            entity.start_token_id = len(tokens_new)

            start = start_init
            end = end_init + 1
            for pieces in pieces_list[start:end]:
                tokens_new += pieces
            entity.end_token_id = len(tokens_new) - 1

            # токены и лейблы сущнсоти
            entity_tokens_new = []
            for t in entity.tokens:
                entity_tokens_new += tokenizer.tokenize(t)

            tag = entity.labels[0].split("-")[-1]
            if len(entity_tokens_new) > 1:
                entity_labels_new = ['B-' + tag] + ['I-' + tag] * (len(entity_tokens_new) - 2) + ['L-' + tag]
            else:
                entity_labels_new = ['U-' + tag]

            entity.tokens = entity_tokens_new
            entity.labels = entity_labels_new

            pointers.append(end_init + 1)

            # print(tokens_new)

        start = pointers[-1]
        for pieces in pieces_list[start:]:
            tokens_new += pieces

        x_new.tokens = tokens_new

    else:
        x_new.tokens = []
        for pieces in pieces_list:
            x_new.tokens += pieces

    return x_new
