import os
import re
import tqdm
from typing import List
from itertools import accumulate
from collections import namedtuple
from rusenttokenize import ru_sent_tokenize

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
            text=None,
            tokens=None,
            labels=None,
            start_index=None,
            end_index=None,
            start_token_id=None,
            end_token_id=None,
            sent_id=None
    ):
        self.id = id
        self.text = text
        self.labels = labels
        self.tokens = tokens
        self.start_index = start_index
        self.end_index = end_index
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.sent_id = sent_id


class Vocab(ReprMixin):
    def __init__(self, values):
        self._id2value = dict(enumerate(sorted(values)))
        self._value2id = {v: k for k, v in self._id2value.items()}

    @property
    def size(self):
        return len(self._id2value)

    def get_value(self, id):
        return self._id2value[id]

    def get_id(self, value):
        return self._value2id[value]


SpanInfo = namedtuple("Span", ["span_chunks", "span_tokens"])


class Example(ReprMixin):
    def __init__(self, id=None, text=None, tokens=None, labels=None, entities=None, arcs=None):
        self.id = id
        self.text = text
        self.tokens = tokens
        self.labels = labels
        self.entities = entities
        self.arcs = arcs

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def num_entities(self):
        return len(self.entities)

    @property
    def chunks(self):
        if not self.text:
            print(f"[{self.id} WARNING]: empty text")
            return [Example(**self.__dict__)]
        sent_candidates = ru_sent_tokenize(self.text)
        lengths = [len(TOKENS_EXPRESSION.findall(sent)) for sent in sent_candidates]
        assert sum(lengths) == len(self.tokens)

        spans = self._get_spans(lengths)

        res = []
        for i, span in enumerate(spans):
            _id = self.id + "_" + str(i)
            start, end = span.span_chunks
            text = ' '.join(sent_candidates[start:end])
            start, end = span.span_tokens
            tokens = self.tokens[start:end]
            labels = self.labels[start:end]
            entities = [x for x in self.entities if start <= x.start_token_id <= x.end_token_id < end]
            entity_ids = {x.id for x in entities}
            arcs = [x for x in self.arcs if (x.head in entity_ids) and (x.dep in entity_ids)]
            x = Example(
                id=_id,
                text=text,
                tokens=tokens,
                labels=labels,
                entities=entities,
                arcs=arcs
            )
            res.append(x)
        return res

    def _get_spans(self, lengths) -> List[SpanInfo]:
        # entity_ptr = 0
        sent_starts = [0] + list(accumulate(lengths))
        spans = []
        start = 0  # начало накопленной последовательности
        start_chunk = 0
        for i in range(len(lengths)):
            is_bad_split = False
            start_next = sent_starts[i + 1]
            start_next_chunk = i + 1
            for entity in self.entities:
                if entity.start_token_id >= start_next:
                    break
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


class ParserRuREBus:
    """
    https://github.com/dialogue-evaluation/RuREBus
    """
    def __init__(self, ner_encoding, ner_label_other="O"):
        assert ner_encoding in {"bio", "bilou"}
        self.ner_encoding = ner_encoding
        self.ner_label_other = ner_label_other

    def parse(self, data_dir, n=None, ner_encoding="bilou"):
        """
        n - сколько примеров распарсить
        """
        assert ner_encoding in {"bio", "bilou"}

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
                        if self.ner_encoding == "bio":
                            if i == 0:
                                prefix = "B"
                            else:
                                prefix = "I"
                        else:
                            if num_entity_tokens == 1:
                                prefix = "U"
                            else:
                                if i == 0:
                                    prefix = "B"
                                elif i == num_entity_tokens - 1:
                                    prefix = "L"
                                else:
                                    prefix = "I"

                        # добавление лейбла
                        label = prefix + "_" + entity_label
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
                        text=actual_entity_pattern,
                        tokens=entity_tokens,
                        labels=entity_labels,
                        start_index=start_index,
                        end_index=end_index,
                        start_token_id=start_token_id,
                        end_token_id=end_token_id,
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
                    arc_triple = arc.head, arc.dep, arc.rel
                    if arc_triple not in arcs_used:
                        arcs.append(arc)
                        arcs_used.add(arc_triple)
                else:
                    raise Exception(f"invalid line: {line}")

        example = Example(
            id=filename,
            text=text,
            tokens=text_tokens,
            labels=ner_labels,
            entities=entities,
            arcs=arcs
        )

        return example


class ExampleEncoder:
    def __init__(self, ner_encoding, ner_label_other="O", re_label_other="O", add_seq_bounds=True):
        assert ner_encoding in {"bio", "bilou"}
        self.ner_encoding = ner_encoding
        self.ner_label_other = ner_label_other
        self.re_label_other = re_label_other
        self.add_seq_bounds = add_seq_bounds

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
                if "_" in label:
                    # предполагаем, что каждая сущность может состоять из нескольких токенов
                    label = label.split("_")[-1]
                    for prefix in prefixes:
                        vocab_ner.add(prefix + "_" + label)
                        vocab_ner.add(prefix + "_" + label)
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

    def transform(self, examples):
        res = []
        for x in examples:
            x_enc = self.transform_example(x)
            res.append(x_enc)
        return res

    def transform_example(self, example):
        """
        Кодирование категориальных атрибутов примеров:
        * tokens - List[str] (остаётся неизменным)
        * labels - List[int]
        * entities - List[Tuple[start, end]]
        * arcs - List[Tuple[head, dep, id_relation]]
        """
        example_enc = Example()

        # tokens
        if self.add_seq_bounds:
            example_enc.tokens = ["[START]"] + example.tokens + ["[END]"]

        # labels
        labels_encoded = []
        for label in example.labels:
            label_enc = self.vocab_ner.get_id(label)
            labels_encoded.append(label_enc)
        if self.add_seq_bounds:
            label = self.vocab_ner.get_id(self.ner_label_other)
            labels_encoded = [label] + labels_encoded + [label]
        example_enc.labels = labels_encoded

        # arcs
        id2index = {x.id: i for i, x in enumerate(sorted(example.entities, key=lambda x: x.start))}
        arcs_encoded = []
        for arc in example.arcs:
            head = id2index[arc.head]
            dep = id2index[arc.dep]
            rel = self.vocab_re.get_id(arc.rel)
            arcs_encoded.append((head, dep, rel))
        example_enc.arcs = arcs_encoded
        return example_enc


def check_example(example: Example, ner_encoding):
    """
    NER:
    * число токенов равно числу лейблов
    * entity.start >= entity.end
    * начало сущности >= 0, конец сущности < len(tokens)
    RE:
    * оба аргумента отношений есть в entities
    """
    assert ner_encoding in {"bio", "bilou"}, f"expected ner_encoding in {{bio, bilou}}, got {ner_encoding}"
    assert len(example.tokens) == len(example.labels), \
        f"[{example.id}] tokens and labels mismatch, {len(example.tokens)} != {len(example.labels)}"

    entity_ids = set()
    for entity in example.entities:
        assert entity.start_token_id <= entity.end_token_id, \
            f"[{example.id}] strange entity span, start = {entity.start_token_id}, end = {entity.end_token_id}"
        # assert entity.start_token_id >= 0, f"[{example.filename}] strange entity start: {entity.start_token_id}"
        # assert entity.end_token_id < len(example.tokens), \
        #     f"[{example.filename}] strange entity end: {entity.end_token_id}, but num tokens is {len(example.tokens)}"
        entity_ids.add(entity.id)

    for arc in example.arcs:
        assert arc.head in entity_ids, \
            f"[{example.id}] something is wrong with arc {arc.id}: head {arc.head} is unknown"
        assert arc.dep in entity_ids, \
            f"[{example.id}] something is wrong with arc {arc.id}: dep {arc.dep} is unknown"

    arcs = [(arc.head, arc.dep, arc.rel) for arc in example.arcs]
    assert len(arcs) == len(set(arcs)), f"[{example.id}] there duplicates in arcs"

    if ner_encoding == "bilou":
        num_start_ids = sum(x.startswith("B") for x in example.labels)
        num_end_ids = sum(x.startswith("L") for x in example.labels)
        assert num_start_ids == num_end_ids, \
            f"[{example.id}]: num start ids: {num_start_ids}, num end ids: {num_end_ids}"
