import os
import re
import tqdm
import json
import shutil
from copy import deepcopy
from typing import List, Set, Dict, Tuple
from itertools import accumulate
from collections import namedtuple, Counter, defaultdict
from rusenttokenize import ru_sent_tokenize


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
    UNDERSCORE = "_"
    HYPHEN = "-"


class LineTypes:
    ENTITY = "T"
    EVENT = "E"
    RELATION = "R"
    ATTRIBUTE = "A"
    # то же самое, что A:
    # https://brat.nlplab.org/standoff.html
    # For backward compatibility with existing standoff formats,
    # brat also recognizes the ID prefix "M" for attributes.
    ATTRIBUTE_OLD = "M"
    COMMENT = "#"


# exceptions

class BadLineException(Exception):
    """
    строка файла .ann имеет неверный формат
    """


class EntitySpanException(Exception):
    """
    спану из файла .ann соответствует другая подстрока в файле .txt
    """


# immutable structs

# possible args: Entity, Event, Arc
Attribute = namedtuple("Attribute", ["id", "type", "value"])
Comment = namedtuple("Comment", ["id", "argument", "text"])
EventArgument = namedtuple("EventArgument", ["id", "role"])
SpanInfo = namedtuple("Span", ["span_chunks", "span_tokens"])


# mutable structs


class ReprMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        params_str = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f'{class_name}({params_str})'


class Entity(ReprMixin):
    def __init__(
            self,
            id=None,
            label=None,
            text=None,
            tokens=None,
            labels=None,
            start_index=None,
            end_index=None,
            start_token_id=None,
            end_token_id=None,
            sent_id=None,
            is_event_trigger=False,
            attrs: Set[Attribute] = None,  # атрибуты сущности
            comment: str = None
    ):
        self.id = id
        self.label = label
        self.text = text
        self.labels = labels
        self.tokens = tokens
        self.start_index = start_index
        self.end_index = end_index
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.sent_id = sent_id
        self.is_event_trigger = is_event_trigger
        self.attrs = attrs if attrs is not None else []
        self.commnet = comment


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


class Example(ReprMixin):
    def __init__(
            self,
            filename=None,
            id=None,
            text=None,
            tokens=None,
            labels=None,
            entities=None,
            arcs=None,
            label=None,
            labels_events: Dict[str, List] = None,
            tokens_spans: List[Tuple[int, int]] = None,
            events: List[Event] = None  # пока только для дебага
    ):
        self.filename = filename
        self.id = id
        self.text = text
        self.tokens = tokens
        self.labels = labels
        self.entities = entities
        self.arcs = arcs
        self.label = label  # в случае классификации предложений
        self.labels_events = labels_events  # NER-лейблы события
        self.tokens_spans = tokens_spans  # нужно для инференса
        self.events = events

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def num_entities(self):
        return len(self.entities)

    @property
    def entities_wo_events(self):
        return [x for x in self.entities if not x.is_event_trigger]

    @property
    def num_entities_wo_events(self):
        return len(self.entities_wo_events)

    @property
    def num_events(self):
        return len(self.events)

    def chunks(self, window=1):
        """
        Кусок исходного примера размером window предложений
        :param window: ширина окна на уровне предложений
        :return:
        """
        if not self.text:
            print(f"[{self.id} WARNING]: empty text")
            return [Example(**self.__dict__)]
        sent_candidates = ru_sent_tokenize(self.text)
        lengths = [len(TOKENS_EXPRESSION.findall(sent)) for sent in sent_candidates]
        assert sum(lengths) == len(self.tokens)

        spans = self._get_spans(lengths, window=window)

        res = []
        for i, span in enumerate(spans):
            _id = self.id + "_" + str(i)
            start, end = span.span_chunks
            text = ' '.join(sent_candidates[start:end])
            start, end = span.span_tokens
            if start == end:
                continue
            tokens = self.tokens[start:end]
            labels = self.labels[start:end]
            tokens_spans = self.tokens_spans[start:end]
            labels_events = {k: v[start:end] for k, v in self.labels_events.items()}
            entities = []
            entity_ids = set()
            for x in self.entities:
                if start <= x.start_token_id <= x.end_token_id < end:
                    x_copy = deepcopy(x)
                    # важно, чтобы не менялись индексы символов начала и конца в исходном документе!
                    x_copy.start_token_id -= start
                    x_copy.end_token_id -= start
                    entities.append(x_copy)
                    entity_ids.add(x.id)
            arcs = [x for x in self.arcs if (x.head in entity_ids) and (x.dep in entity_ids)]
            x = Example(
                filename=self.filename,
                id=_id,
                text=text,
                tokens=tokens,
                labels=labels,
                entities=entities,
                arcs=arcs,
                labels_events=labels_events,
                tokens_spans=tokens_spans
            )
            res.append(x)
        return res

    def _get_spans(self, lengths, window=1) -> List[SpanInfo]:
        # entity_ptr = 0
        sent_starts = [0] + list(accumulate(lengths))
        spans = []
        start = 0  # начало накопленной последовательности
        start_chunk = 0
        for i in range(len(lengths) - window + 1):
            # разделение на предложения плохое, если оно проходит через именную сущность
            is_bad_split = False
            start_next = sent_starts[i + window]
            start_next_chunk = i + window
            # TODO: можно оптимизировать, бегая каждый раз не по всем сущностям
            for entity in self.entities:
                # это условие имеет есто, если сущности сортированы по спану:
                # if entity.start_token_id >= start_next:
                #     break
                # начало находится в куске i, конец - в куске i + 1
                if entity.start_token_id < start_next <= entity.end_token_id:
                    is_bad_split = True
                    break
            if not is_bad_split:
                span_chunks = start_chunk, start_next_chunk
                span_tokens = start, start_next
                span = SpanInfo(span_chunks=span_chunks, span_tokens=span_tokens)
                spans.append(span)
                start = start_next
                start_chunk = i + 1
        return spans


# handlers


class ExamplesLoader:
    """
    * загрузка примеров из brat-формата: пар (filename.txt, filename.ann)
    * сохранение примров brat-формат: файлы filename.ann

    примеры датасетов в таком (brat) формате:
    https://github.com/dialogue-evaluation/RuREBus

    """

    def __init__(
            self,
            ner_encoding: str = NerEncodings.BILOU,
            ner_label_other: str = "O",
            ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
            fix_new_line_symbol: bool = True,
            event_tags: Set[str] = None
    ):
        assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}
        self.ner_encoding = ner_encoding
        self.ner_label_other = ner_label_other
        self.ner_prefix_joiner = ner_prefix_joiner
        self.fix_new_line_symbol = fix_new_line_symbol
        self.event_tags = event_tags if event_tags is not None else set()  # таги событий

    def load_examples(
            self,
            data_dir: str,
            n: int = None,
            split: bool = True,
            window: int = 1
    ) -> List[Example]:
        examples = []
        num_bad = 0
        num_examples = 0
        for x_raw in self.parse(data_dir=data_dir, n=n):
            # проверяем целый пример
            try:
                self.check_example(example=x_raw)
            except AssertionError as e:
                print("[doc]", e)
                num_bad += 1
                continue

            if split:
                for x_raw_chunk in x_raw.chunks(window=window):
                    num_examples += 1
                    # проверяем кусок примера
                    try:
                        self.check_example(example=x_raw_chunk)
                        examples.append(x_raw_chunk)
                    except AssertionError as e:
                        print("[sent]", e)
                        num_bad += 1
            else:
                num_examples += 1
                examples.append(x_raw)
        print(f"{num_bad} / {num_examples} examples are bad")
        return examples

    @staticmethod
    def save_predictions(
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

    def check_example(self, example: Example):
        """
        NER:
        * число токенов равно числу лейблов
        * entity.start >= entity.end
        * начало сущности >= 0, конец сущности < len(tokens)
        RE:
        * оба аргумента отношений есть в entities
        """
        # обязателен айдишник
        assert example.id is not None, f"example {example} has no id!"
        prefix = f"[{example.id}]: "

        # ner-кодировка
        assert self.ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}, \
            f"expected ner_encoding {NerEncodings.BIO} or {NerEncodings.BILOU}, got {self.ner_encoding}"

        num_tokens = len(example.tokens)

        # биекция между токенами и лейблами
        assert num_tokens == len(example.labels), \
            prefix + f"tokens and labels mismatch, {num_tokens} != {len(example.labels)}"

        entity_ids_all = set()
        entity_ids_wo_events = set()
        entity_spans = set()
        event2entities = defaultdict(set)
        for entity in example.entities:
            # обязателен айдишник
            assert entity.id is not None, \
                prefix + f"[{entity}] entity has no id!"

            # проверка валидности спана
            assert 0 <= entity.start_token_id <= entity.end_token_id < num_tokens, \
                prefix + f"[{entity}] strange entity span: " \
                    f"start token id: {entity.start_token_id}, end token id: {entity.end_token_id}. num tokens: {num_tokens}"

            # проверка корректности соответстия токенов сущности токенам примера
            expected_tokens = example.tokens[entity.start_token_id:entity.end_token_id + 1]
            assert expected_tokens == entity.tokens, \
                prefix + f"[{entity}] tokens and example tokens mismatch: {entity.tokens} != {expected_tokens}"

            # проверка корректности соответстия лейблов сущности лейблам примера
            if entity.is_event_trigger:
                ner_labels = example.labels_events[entity.label]
            else:
                ner_labels = example.labels
            expected_labels = ner_labels[entity.start_token_id:entity.end_token_id + 1]
            assert expected_labels == entity.labels, \
                prefix + f"[{entity}]: labels and example labels mismatch: {entity.labels} != {expected_labels}"

            # кэш
            entity_ids_all.add(entity.id)
            entity_spans.add((entity.start_token_id, entity.end_token_id))
            if entity.is_event_trigger:
                event2entities[entity.label].add(entity.id)
            else:
                entity_ids_wo_events.add(entity.id)

        # проверка уникальности сущностей
        assert len(example.entities) == len(entity_ids_all), \
            prefix + f"entity ids are not unique: {len(example.entities)} != {len(entity_ids_all)}"

        # проверка биекции между множеством спанов и множеством сущностей.
        # пока предполагается её наличие.
        assert len(example.entities) == len(entity_spans), \
            prefix + f"there are span duplicates: " \
            f"number of entities is {len(example.entities)}, but number of unique text spans is {len(entity_spans)}"

        def check_ner_labels(ent_ids, labels, ner_label_other):
            """проверка непротиворечивости множества сущностей лейблам"""
            if len(ent_ids) == 0:
                assert set(labels) == {ner_label_other}, \
                    prefix + f"ner labels and named entities mismatch: ner labels are {set(labels)}, " \
                    f"but there are no entities in example."
            else:
                assert set(labels) != {ner_label_other}, \
                    prefix + f"ner labels and named entities mismatch: ner labels are {set(labels)}, " \
                    f"but there are following entities in example: {ent_ids}"

        check_ner_labels(ent_ids=entity_ids_wo_events, labels=example.labels, ner_label_other=self.ner_label_other)

        for k, v in event2entities.items():
            check_ner_labels(ent_ids=v, labels=example.labels_events[k], ner_label_other=self.ner_label_other)

        arc_args = []
        for arc in example.arcs:
            # проверка того, что в примере есть исходящая вершина
            assert arc.head in entity_ids_all, \
                prefix + f"something is wrong with arc {arc.id}: head {arc.head} is unknown"
            # проверка того, что в примере есть входящая вершина
            assert arc.dep in entity_ids_all, \
                prefix + f"something is wrong with arc {arc.id}: dep {arc.dep} is unknown"
            arc_args.append((arc.head, arc.dep))
        # проверка того, что одному ребру соответствует одно отношение
        if len(arc_args) != len(set(arc_args)):
            arc_counts = {k: v for k, v in Counter(arc_args).items() if v > 1}
            raise AssertionError(prefix + f'there duplicates in arc args: {arc_counts}')

        if self.ner_encoding == NerEncodings.BILOU:
            # проверка того, что число начал сущности равно числу концов
            num_start_ids = sum(x.startswith("B") for x in example.labels)
            num_end_ids = sum(x.startswith("L") for x in example.labels)
            assert num_start_ids == num_end_ids, \
                prefix + f"num start ids: {num_start_ids}, num end ids: {num_end_ids}"

    def parse(self, data_dir, n=None, ner_encoding=NerEncodings.BILOU):
        """
        n - сколько примеров распарсить
        """
        assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}

        # выбираем файлы, для которых есть исходный текст и разметка
        files = os.listdir(data_dir)
        texts = {x.split('.')[0] for x in files if x.endswith('.txt')}
        answers = {x.split('.')[0] for x in files if x.endswith('.ann')}
        names_to_use = sorted(texts & answers)  # сортировка для детерминированности
        print(f"num docs: {len(names_to_use)}")

        # парсим примеры для обучения
        examples = []
        for filename in tqdm.tqdm(names_to_use[:n]):
            try:
                example = self._parse_example(data_dir=data_dir, filename=filename)
                examples.append(example)
            except BadLineException as e:
                print(e)
            except EntitySpanException as e:
                print(e)
            except Exception as e:
                print(e)
        print(f"num parsed examples: {len(examples)}")
        return examples

    def _parse_example(self, data_dir, filename: str):
        """
        строчка файла filename.ann:

        * сущность: T5\tBIN 325 337\tФормирование\n
        * отношение: R105\tTSK Arg1:T370 Arg2:T371\n
        * событие: E0\tBankruptcy:T0 Bankrupt:T1 Bankrupt2:T2\n
        * атрибут: A1\tEvent-time E0 Past\n
        * комментарий: #0\tAnnotatorNotes T3\tfoobar\n

        замечения:
        * в файлах .ann сначала пишется триггер события, а потом событие:
        T12     Bankruptcy 1866 1877    банкротства
        E3      Bankruptcy:T12
        * в файлах .ann аргумент атрибута всегда указан раньше, чем сам атрибут:
        E3      Bankruptcy:T12
        A10     Negation E3

        данные наблюдения позволяют за один проход по всем строчка файла .ann сделать всё необходимое
        """
        # подгрузка текста
        with open(os.path.join(data_dir, f'{filename}.txt')) as f:
            text = ' '.join(f)
            if self.fix_new_line_symbol:
                text = text.replace('\n ', '\n')

        # токенизация
        text_tokens = []
        tokens_spans = []
        span2index = {}

        # бывают странные ситуации:
        # @ подстрока текста: передачи данных___________________7;
        # @ в файле .ann есть сущность "данных"
        # @ TOKENS_EXPRESSION разбивает на токены так: [передачи, данных___________________7]
        # @ получается невозможно определить индекс токена "данных"
        # @ будем в таком случае пытаться это сделать по индексу начала
        start2index = {}
        for i, m in enumerate(TOKENS_EXPRESSION.finditer(text)):
            text_tokens.append(m.group())

            span = m.span()
            span2index[span] = i
            start2index[span[0]] = i
            tokens_spans.append(span)

        # .ann
        ner_labels = [self.ner_label_other] * len(text_tokens)
        ner_labels_events = {event_tag: ner_labels.copy() for event_tag in self.event_tags}
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
                        raise BadLineException(f"[{filename}]: something is wrong with line: {line}")

                    entity_label, start_index, end_index = entity.split()
                    start_index = int(start_index)
                    end_index = int(end_index)

                    # проверка того, что в файле .txt в спане из файла .ann находится
                    # правильная именная сущность
                    actual_entity_pattern = text[start_index:end_index]
                    if actual_entity_pattern != expected_entity_pattern:
                        raise EntitySpanException(f"[{filename}]: something is wrong with markup; "
                                                  f"expected entity is {expected_entity_pattern}, "
                                                  f"but got {actual_entity_pattern}")

                    entity_matches = list(TOKENS_EXPRESSION.finditer(expected_entity_pattern))
                    start_token_id = None
                    end_token_id = None
                    entity_labels = []
                    entity_tokens = []
                    num_entity_tokens = len(entity_matches)
                    for i, m in enumerate(entity_matches):
                        # добавление токена сущности
                        token = m.group()
                        entity_tokens.append(token)

                        # вывод префикса:
                        if self.ner_encoding == NerEncodings.BIO:
                            if i == 0:
                                prefix = "B"
                            else:
                                prefix = "I"
                        elif self.ner_encoding == NerEncodings.BILOU:
                            if num_entity_tokens == 1:
                                prefix = "U"
                            else:
                                if i == 0:
                                    prefix = "B"
                                elif i == num_entity_tokens - 1:
                                    prefix = "L"
                                else:
                                    prefix = "I"
                        else:
                            raise

                        # добавление лейбла
                        label = prefix + self.ner_prefix_joiner + entity_label
                        entity_labels.append(label)

                        # вывод спана токена в исходном тексте
                        si, ei = m.span()
                        token_span_abs = start_index + si, start_index + ei

                        try:
                            # вывод порядкового номера токена
                            # выполненное условие actual_entity_pattern == text[start_index:end_index]
                            # гарантирует отсутствие KeyError здесь:
                            token_id = span2index[token_span_abs]
                        except KeyError as e:
                            print("***** WARNING: can not infer token id from absolute span! *****")
                            print("file:", filename)
                            print("absolute span:", token_span_abs)
                            print("entity token:", token)
                            print("corresponding text token:", text[token_span_abs[0]:token_span_abs[1]])
                            print("context:", text[token_span_abs[0] - 50:token_span_abs[1] + 50])
                            try:
                                token_id = start2index[token_span_abs[0]]
                            except KeyError:
                                print("***** ERROR: can not infer token id from start index! *****")
                                print("file:", filename)
                                print("absolute span:", token_span_abs)
                                print("entity token:", token)
                                print("corresponding text token:", text[token_span_abs[0]:token_span_abs[1]])
                                print("context:", text[token_span_abs[0] - 50:token_span_abs[1] + 50])
                                raise e

                        assert token_id is not None

                        # запись лейблов
                        if entity_label in self.event_tags:
                            ner_labels_events[entity_label][token_id] = label
                        else:
                            ner_labels[token_id] = label

                        # вывод индекса токена начала и конца
                        if i == 0:
                            start_token_id = token_id
                        if i == num_entity_tokens - 1:
                            end_token_id = token_id

                    assert start_token_id is not None
                    assert end_token_id is not None
                    assert TOKENS_EXPRESSION.findall(actual_entity_pattern) == entity_tokens

                    # создание сущности
                    entity = Entity(
                        id=line_tag,
                        label=entity_label,
                        text=actual_entity_pattern,
                        tokens=entity_tokens,
                        labels=entity_labels,
                        start_index=start_index,
                        end_index=end_index,
                        start_token_id=start_token_id,
                        end_token_id=end_token_id,
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
                        raise BadLineException(f"[{filename}]: something is wrong with line: {line}")
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
                    event_name, id_head = event_trigger.split(":")  # event_name не используется
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
                    id2entity[id_head].is_event_trigger = True

                # атрибут
                elif line_type == LineTypes.ATTRIBUTE or line_type == LineTypes.ATTRIBUTE_OLD:
                    # A1\tEvent-time E0 Past\n
                    attr_type, id_arg, value = content[1].split()
                    attr = Attribute(id=line_tag, type=attr_type, value=value)
                    try:
                        id2arg[id_arg].attrs.append(attr)
                    except KeyError:
                        raise BadLineException("there is no arg for provided attr")

                # комментарии.
                elif line_type == LineTypes.COMMENT:
                    # #0\tAnnotatorNotes T3\tfoobar\n
                    _, id_arg = content[1].split()
                    msg = content[2]
                    try:
                        id2arg[id_arg].comment = msg
                    except KeyError:
                        raise BadLineException("there is no arg for provided comment")

                else:
                    raise Exception(f"invalid line: {line}")

        entities = list(id2entity.values())
        events = list(id2event.values())
        arcs = list(id2arc.values())

        example = Example(
            filename=filename,
            id=filename,
            text=text,
            tokens=text_tokens,
            labels=ner_labels,
            entities=entities,
            arcs=arcs,
            labels_events=ner_labels_events,
            tokens_spans=tokens_spans,
            events=events
        )

        return example


class ExampleEncoder:
    def __init__(
            self,
            ner_encoding: str = NerEncodings.BIO,
            ner_label_other: str = "O",
            re_label_other: str = "O",
            ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
            add_seq_bounds: bool = True
    ):
        assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}
        self.ner_encoding = ner_encoding
        self.ner_label_other = ner_label_other
        self.re_label_other = re_label_other
        self.add_seq_bounds = add_seq_bounds
        self.ner_prefix_joiner = ner_prefix_joiner

        self.vocab_ner = None
        self.vocab_re = None
        self.vocabs_events = {}

    def fit_transform(self, examples):
        self.fit(examples)
        return self.transform(examples)

    def fit(self, examples):
        # инициализация значений словаря
        vocab_ner = set()
        vocabs_events = defaultdict(set)

        prefixes = {"B", "I"}
        if self.ner_encoding == NerEncodings.BILOU:
            prefixes |= {"L", "U"}

        def extend_vocab(label_, ner_prefix_joiner, vocab_values):
            if ner_prefix_joiner in label_:
                # предполагаем, что каждая сущность может состоять из нескольких токенов
                label_ = label_.split(ner_prefix_joiner)[-1]
                for p in prefixes:
                    vocab_values.add(p + ner_prefix_joiner + label_)
            else:
                vocab_values.add(label_)

        for x in examples:
            for label in x.labels:
                extend_vocab(label, self.ner_prefix_joiner, vocab_ner)
            for event_tag, labels in x.labels_events.items():
                for label in labels:
                    extend_vocab(label, self.ner_prefix_joiner, vocabs_events[event_tag])

        vocab_ner.add(self.ner_label_other)
        self.vocab_ner = Vocab(vocab_ner)

        self.vocabs_events = {}
        for k, v in vocabs_events.items():
            v.add(self.ner_label_other)
            self.vocabs_events[k] = Vocab(v)

        # arcs vocab
        vocab_re = set()
        for x in examples:
            for arc in x.arcs:
                vocab_re.add(arc.rel)
        vocab_re.add(self.re_label_other)
        self.vocab_re = Vocab(vocab_re)

    def transform(self, examples: List[Example]) -> List[Example]:
        res = []
        for x in examples:
            x_enc = self.transform_example(x)
            res.append(x_enc)
        return res

    def transform_example(self, example: Example) -> Example:
        """
        Кодирование категориальных атрибутов примеров:
        * tokens - List[str] (остаётся неизменным)
        * labels - List[int]
        * entities - List[Tuple[start, end]]
        * arcs - List[Tuple[head, dep, id_relation]]
        """
        example_enc = Example(
            filename=example.filename,
            id=example.id,
            text=example.text
        )

        # tokens
        example_enc.tokens = example.tokens.copy()
        if self.add_seq_bounds:
            example_enc.tokens = ["[START]"] + example_enc.tokens + ["[END]"]

        # tokens spans
        example_enc.tokens_spans = example.tokens_spans.copy()
        if self.add_seq_bounds:
            example_enc.tokens_spans = [(-1, -1)] + example_enc.tokens_spans + [(-1, -1)]  # TODO: ок ли так делать?

        # labels
        def encode_labels(labels, vocab, add_seq_bounds, ner_label_other):
            labels_encoded = []
            for label in labels:
                label_enc = vocab.get_id(label)
                labels_encoded.append(label_enc)
            if add_seq_bounds:
                label = vocab.get_id(ner_label_other)
                labels_encoded = [label] + labels_encoded + [label]
            # example_enc.labels = labels_encoded
            return labels_encoded

        example_enc.labels = encode_labels(
            labels=example.labels, vocab=self.vocab_ner,
            add_seq_bounds=self.add_seq_bounds, ner_label_other=self.ner_label_other
        )
        example_enc.labels_events = {}
        for k, v in example.labels_events.items():
            example_enc.labels_events[k] = encode_labels(
                labels=v, vocab=self.vocabs_events[k],
                add_seq_bounds=self.add_seq_bounds, ner_label_other=self.ner_label_other
            )

        # entities
        example_enc.entities = deepcopy(example.entities)
        if self.add_seq_bounds:
            # потому что в начало добавлен токен начала строки
            for entity in example_enc.entities:
                entity.start_token_id += 1
                entity.end_token_id += 1

        # arcs
        arcs_encoded = []
        for arc in example.arcs:
            id_rel = self.vocab_re.get_id(arc.rel)
            arc_enc = Arc(id=arc.id, head=arc.head, dep=arc.dep, rel=id_rel)
            arcs_encoded.append(arc_enc)
        example_enc.arcs = arcs_encoded
        return example_enc

    def save(self, encoder_dir):
        d = {
            "ner_encoding": self.ner_encoding,
            "ner_label_other": self.ner_label_other,
            "re_label_other": self.re_label_other,
            "ner_prefix_joiner": self.ner_prefix_joiner,
            "add_seq_bounds": self.add_seq_bounds
        }
        with open(os.path.join(encoder_dir, "encoder_config.json"), "w") as f:
            json.dump(d, f, indent=4)

        with open(os.path.join(encoder_dir, "ner_encodings.json"), "w") as f:
            json.dump(self.vocab_ner.encodings, f, indent=4)

        with open(os.path.join(encoder_dir, "ner_encodings_events.json"), "w") as f:
            json.dump({k: v.encodings for k, v in self.vocabs_events.items()}, f, indent=4)

        with open(os.path.join(encoder_dir, "re_encodings.json"), "w") as f:
            json.dump(self.vocab_re.encodings, f, indent=4)

    @classmethod
    def load(cls, encoder_dir):
        config = json.load(open(os.path.join(encoder_dir, "encoder_config.json")))
        enc = cls(**config)

        ner_encodings = json.load(open(os.path.join(encoder_dir, "ner_encodings.json")))
        enc.vocab_ner = Vocab(values=ner_encodings)

        re_encodings = json.load(open(os.path.join(encoder_dir, "re_encodings.json")))
        enc.vocab_re = Vocab(values=re_encodings)

        d = json.load(open(os.path.join(encoder_dir, "ner_encodings_events.json")))
        enc.vocabs_events = {k: Vocab(values=v) for k, v in d.items()}

        return enc


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
