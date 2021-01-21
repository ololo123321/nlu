import os
import re
import tqdm
import json
from copy import deepcopy
from typing import List, Set, Dict
from itertools import accumulate
from collections import namedtuple, Counter, defaultdict
from rusenttokenize import ru_sent_tokenize

from .utils import SpecialSymbols, BertEncodings, NerEncodings


TOKENS_EXPRESSION = re.compile(r"\w+|[^\w\s]")


# TOKENS_EXPRESSION = re.compile("|".join([  # порядок выражений важен!
#     r"[А-ЯA-Z]\w*[\.-]?\w+",  # Foo.bar -> Foo.Bar; Foo.bar -> Foo.bar
#     r"[а-яa-z]\w*[\.-]?[а-яa-z]\w*",  # foo.bar -> foo.bar
#     r"\w+",  # слова, числа
#     r"[^\w\s]"  # пунктуация
# ]))


class BadLineException(Exception):
    """
    строка файла .ann имеет неверный формат
    """


class EntitySpanException(Exception):
    """
    спану из файла .ann соответствует другая подстрока в файле .txt
    """


class ReprMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        params_str = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f'{class_name}({params_str})'


class Arc(ReprMixin):
    def __init__(self, id, head, dep, rel):
        self.id = id
        self.head = head
        self.dep = dep
        self.rel = rel


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


EventArgument = namedtuple("EventArgument", ["id", "role"])


class Event:
    def __init__(self, id=None, id_trigger=None, label=None, arguments: Set[EventArgument] = None):
        self.id_trigger = id_trigger
        self.label = label
        self.arguments = arguments if arguments is not None else set()
        self.id = id


class Vocab(ReprMixin):
    def __init__(self, values):
        if isinstance(values, dict):  # str -> int
            self._value2id = values
            self._id2value = {v: k for k, v in values.items()}
        else:
            special_value = "O"
            values -= {special_value}
            self._id2value = dict(enumerate(sorted(values), 1))
            self._id2value[0] = special_value
            self._value2id = {v: k for k, v in self._id2value.items()}

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


SpanInfo = namedtuple("Span", ["span_chunks", "span_tokens"])


class Example(ReprMixin):
    def __init__(self, filename=None, id=None, text=None, tokens=None, labels=None, entities=None, arcs=None, label=None):
        self.filename = filename
        self.id = id
        self.text = text
        self.tokens = tokens
        self.labels = labels
        self.entities = entities
        self.arcs = arcs
        self.label = label  # в случае классификации предложений

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def num_entities(self):
        return len(self.entities)

    def chunks(self, window=1):
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
                arcs=arcs
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
            ner_suffix_joiner: str = '-'
    ):
        assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}
        self.ner_encoding = ner_encoding
        self.ner_label_other = ner_label_other
        self.ner_suffix_joiner = ner_suffix_joiner

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
        print(f"{num_bad} / {len(examples)} examples are bad")
        return examples

    @staticmethod
    def save_predictions(
            examples: List[Example],
            output_dir: str,
            id2relation: Dict[int, str]
    ):
        event_counter = defaultdict(int)
        for x in examples:
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
                                id_trigger=entity.id,
                                label=entity.label,
                                arguments=None,
                            )
                            event_counter[x.filename] += 1

                # отношения
                for arc in x.arcs:
                    rel = id2relation[arc.rel]
                    if arc.head in events:
                        arg = EventArgument(id=arc.dep, role=rel)
                        events[arc.head].arguments.add(arg)
                    else:
                        line = f"R{arc.id}\t{rel} Arg1:{arc.head} Arg2:{arc.dep}\n"
                        f.write(line)

                # события
                for event in events.values():
                    line = f"E{event.id}\t{event.label}:{event.id_trigger}"
                    role2count = defaultdict(int)
                    args_str = ""
                    for arg in event.arguments:
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

    def check_example(self, example: Example):
        """
        NER:
        * число токенов равно числу лейблов
        * entity.start >= entity.end
        * начало сущности >= 0, конец сущности < len(tokens)
        RE:
        * оба аргумента отношений есть в entities
        """
        assert example.id is not None, f"example {example} has no id!"
        prefix = f"[{example.id}]: "

        assert self.ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}, \
            f"expected ner_encoding {NerEncodings.BIO} or {NerEncodings.BILOU}, got {self.ner_encoding}"
        assert len(example.tokens) == len(example.labels), \
            prefix + f"tokens and labels mismatch, {len(example.tokens)} != {len(example.labels)}"

        entity_ids = set()
        entity_spans = set()
        for entity in example.entities:
            assert entity.id is not None, \
                prefix + f"[{entity}] entity has no id!"
            assert entity.start_token_id <= entity.end_token_id, \
                prefix + f"[{entity}] strange entity span"
            assert entity.start_token_id >= 0, prefix + f"[{entity}] strange start"
            assert entity.end_token_id < len(example.tokens), \
                prefix + f"[{entity}] strange entity end: {entity.end_token_id}, but num tokens is {len(example.tokens)}"
            expected_tokens = example.tokens[entity.start_token_id:entity.end_token_id + 1]
            assert expected_tokens == entity.tokens, \
                prefix + f"[{entity}] tokens and example tokens mismatch: {entity.tokens} != {expected_tokens}"
            expected_labels = example.labels[entity.start_token_id:entity.end_token_id + 1]
            assert expected_labels == entity.labels, \
                prefix + f"[{entity}]: labels and example labels mismatch: {entity.labels} != {expected_labels}"
            entity_ids.add(entity.id)
            entity_spans.add((entity.start_token_id, entity.end_token_id))

        assert len(example.entities) == len(entity_ids), \
            prefix + f"entity ids are not unique: {len(example.entities)} != {len(entity_ids)}"

        assert len(example.entities) == len(entity_spans), \
            prefix + f"there are overlapping entitie: " \
                f"number of entities is {len(example.entities)}, but number of unique text spans is {len(entity_spans)}"

        if len(entity_ids) == 0:
            assert set(example.labels) == {self.ner_label_other}, \
                prefix + f"ner labels and named entities mismatch: ner labels are {set(example.labels)}, " \
                    f"but there are no entities in example."
        else:
            assert set(example.labels) != {self.ner_label_other}, \
                prefix + f"ner labels and named entities mismatch: ner labels are {set(example.labels)}, " \
                    f"but there is at least one entity in example."

        arc_args = []
        for arc in example.arcs:
            assert arc.head in entity_ids, \
                prefix + f"something is wrong with arc {arc.id}: head {arc.head} is unknown"
            assert arc.dep in entity_ids, \
                prefix + f"something is wrong with arc {arc.id}: dep {arc.dep} is unknown"
            arc_args.append((arc.head, arc.dep))
        if len(arc_args) != len(set(arc_args)):
            arc_counts = {k: v for k, v in Counter(arc_args).items() if v > 1}
            raise AssertionError(prefix + f'there duplicates in arc args: {arc_counts}')

        if self.ner_encoding == NerEncodings.BILOU:
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
        строчка файла filename:
        сущность:
        T5\tBIN 325 337\tФормирование\n
        отношение:
        R105\tTSK Arg1:T370 Arg2:T371
        """
        # подгрузка текста
        with open(os.path.join(data_dir, f'{filename}.txt')) as f:
            text = ' '.join(f)
            text = text.replace('\n ', '\n')

        # токенизация
        text_tokens = []
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
            span2index[m.span()] = i
            start2index[m.span()[0]] = i

        # .ann
        ner_labels = [self.ner_label_other] * len(text_tokens)
        entities = []
        arcs = []
        arcs_used = set()  # в арках бывают дубликаты, пример: R35, R36 в 31339011023601075299026_18_part_1.ann
        event_triggers = set()
        with open(os.path.join(data_dir, f'{filename}.ann'), 'r') as f:
            for line in f:
                line = line.strip()
                content = line.split('\t')
                line_tag = content[0]
                if line_tag.startswith("T"):
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
                        label = prefix + self.ner_suffix_joiner + entity_label
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

                        # запись лейблов в ner_labels
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
                    entities.append(entity)

                elif line_tag.startswith("R"):
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
                    # arc_triple = arc.head, arc.dep, arc.rel
                    arc_triple = arc.head, arc.dep
                    if arc_triple not in arcs_used:
                        arcs.append(arc)
                        arcs_used.add(arc_triple)

                elif line_tag.startswith("E"):
                    # E0\tBankruptcy:T0 Bankrupt:T1 Bankrupt2:T2
                    event_args = content[1].split()
                    event_trigger = event_args.pop(0)
                    event_name, id_head = event_trigger.split(":")  # TODO: event_name не используется
                    event_triggers.add(id_head)
                    for dep in event_args:
                        rel, id_dep = dep.split(":")
                        rel = re.sub(r'\d+', '', rel)  # если аргументов несколько, то 
                        arc = Arc(
                            id=line_tag,
                            head=id_head,
                            dep=id_dep,
                            rel=rel
                        )
                        arc_triple = arc.head, arc.dep
                        if arc_triple not in arcs_used:
                            arcs.append(arc)
                            arcs_used.add(arc_triple)
                else:
                    raise Exception(f"invalid line: {line}")

        for entity in entities:
            entity.is_event_trigger = entity.id in event_triggers

        example = Example(
            filename=filename,
            id=filename,
            text=text,
            tokens=text_tokens,
            labels=ner_labels,
            entities=entities,
            arcs=arcs
        )

        return example


class ExampleEncoder:
    def __init__(
            self,
            ner_encoding,
            ner_label_other="O",
            re_label_other="O",
            ner_suffix_joiner='_',
            add_seq_bounds=True
    ):
        assert ner_encoding in {"bio", "bilou"}
        self.ner_encoding = ner_encoding
        self.ner_label_other = ner_label_other
        self.re_label_other = re_label_other
        self.add_seq_bounds = add_seq_bounds
        self.ner_suffix_joiner = ner_suffix_joiner

        self.vocab_ner = None
        self.vocab_re = None

    def fit_transform(self, examples):
        self.fit(examples)
        return self.transform(examples)

    def fit(self, examples):
        vocab_ner = set()
        prefixes = {"B", "I"}
        if self.ner_encoding == "bilou":
            prefixes |= {"L", "U"}
        for x in examples:
            for label in x.labels:
                if self.ner_suffix_joiner in label:
                    # предполагаем, что каждая сущность может состоять из нескольких токенов
                    label = label.split(self.ner_suffix_joiner)[-1]
                    for prefix in prefixes:
                        vocab_ner.add(prefix + self.ner_suffix_joiner + label)
                        vocab_ner.add(prefix + self.ner_suffix_joiner + label)
                else:
                    vocab_ner.add(label)
        vocab_ner.add(self.ner_label_other)
        self.vocab_ner = Vocab(vocab_ner)

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
        example_enc = Example(filename=example.filename, id=example.id)

        # tokens
        example_enc.tokens = example.tokens.copy()
        if self.add_seq_bounds:
            example_enc.tokens = ["[START]"] + example_enc.tokens + ["[END]"]

        # labels
        labels_encoded = []
        for label in example.labels:
            label_enc = self.vocab_ner.get_id(label)
            labels_encoded.append(label_enc)
        if self.add_seq_bounds:
            label = self.vocab_ner.get_id(self.ner_label_other)
            labels_encoded = [label] + labels_encoded + [label]
        example_enc.labels = labels_encoded

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
            "ner_suffix_joiner": self.ner_suffix_joiner,
            "add_seq_bounds": self.add_seq_bounds
        }
        with open(os.path.join(encoder_dir, "encoder_config.json"), "w") as f:
            json.dump(d, f, indent=4)
        
        with open(os.path.join(encoder_dir, "ner_encodings.json"), "w") as f:
            json.dump(self.vocab_ner.encodings, f, indent=4)
        
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
