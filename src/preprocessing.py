import os
import re
import tqdm
from collections import defaultdict
from copy import deepcopy
from typing import List

from rusenttokenize import ru_sent_tokenize


TOKENS_EXPRESSION = re.compile("|".join([  # порядок выражений важен!
    r"[А-ЯA-Z]\w*[\.-]?\w+",  # Foo.bar -> Foo.Bar; Foo.bar -> Foo.bar
    r"[а-яa-z]\w*[\.-]?[а-яa-z]\w*",  # foo.bar -> foo.bar
    r"\w+",  # слова, числа
    r"[^\w\s]"  # пунктуация
]))


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
    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end


class NERLabel:
    def __init__(self, id_entity, value):
        self.id_entity = id_entity
        self.value = value


class Vocab:
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


class Example(ReprMixin):
    def __init__(self, filename, text, tokens, labels, entities, arcs):
        """
        entities[i] = (start, end)
        arcs[j] = (id_head, id_dep, rel)

        по идее в arcs здесь должны быть вообще все возможные пары (id_head, id_dep).
        и если для пары (id_head, id_dep) отношение неизвестно, то нужно ему явно присвоить айдишник
        неизвестного отношения (как в случае с лейблом "O" в NER)
        """
        self.filename = filename
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


class ParserRuREBus:
    """
    https://github.com/dialogue-evaluation/RuREBus
    """
    NER_LABEL_OTHER = '0'
    RE_LABEL_OTHER = 'O'

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def parse(self, n=None, ner_encoding="bilou"):
        """
        n - сколько примеров распарсить
        """
        assert ner_encoding in {"bio", "bilou"}

        # выбираем файлы, для которых есть исходный текст и разметка
        files = os.listdir(self.data_dir)
        texts = {x.split('.')[0] for x in files if x.endswith('.txt')}
        answers = {x.split('.')[0] for x in files if x.endswith('.ann')}
        names_to_use = sorted(texts & answers)  # сортировка для детерминированности
        print(f"num docs: {len(names_to_use)}")

        # парсим примеры для обучения
        examples = []
        for filename in tqdm.tqdm(names_to_use[:n]):
            example = self._parse_example(filename, ner_encoding=ner_encoding)
            if example is not None:
                examples.append(example)
        print(f"num parsed examples: {len(examples)}")
        return examples

    @staticmethod
    def split_example(example: Example, window: int = 1):
        """
        Если example построен на уровне документа, то нужно разбить его
        на куски размера window предложений.
        """
        sentences = ru_sent_tokenize(example.text)
        lengths = [len(TOKENS_EXPRESSION.findall(s)) for s in sentences]
        assert sum(lengths) == len(example.tokens)

        examples = []

        # Entity -> Set[Arc]
        id_entity2arcs = defaultdict(set)

        for arc in example.arcs:
            id_entity2arcs[arc.head].add(arc)
            id_entity2arcs[arc.dep].add(arc)

        for id_sent in range(0, len(lengths) - window + 1):
            start = sum(lengths[:id_sent])
            end = start + sum(lengths[id_sent:id_sent + window])

            tokens_new = example.tokens[start:end]
            labels_new = example.labels[start:end]

            # нужно взять только те сущности entity, что:
            # * start <= entity.start <= end
            # * start <= entity.end <= end
            entities_new = []
            arcs_new = {}
            for i_old, entity in enumerate(example.entities):
                # [      (            )          ]
                # start  start_entity end_entity end
                if start <= entity.start and entity.end < end:
                    start_entity_new = entity.start - start
                    end_entity_new = entity.end - start
                    entity_new = Entity(id=entity.id, start=start_entity_new, end=end_entity_new)
                    entities_new.append(entity_new)

                    for arc in id_entity2arcs[entity.id]:
                        arcs_new[arc.id] = deepcopy(arc)
            arcs_new = list(arcs_new.values())

            text_new = ' '.join(sentences[id_sent:id_sent + window])
            x_new = Example(
                filename=example.filename,
                text=text_new,
                tokens=tokens_new,
                labels=labels_new,
                entities=entities_new,
                arcs=arcs_new,
            )
            examples.append(x_new)

        return examples

    @classmethod
    def encode_example(cls, example: Example, vocab_ner: Vocab, vocab_re: Vocab, add_bounds: bool = False):
        """
        Кодирование категориальных атрибутов примеров:
        * tokens - List[str] (остаётся неизменным)
        * labels - List[int]
        * entities - List[Tuple[start, end]]
        * arcs - List[Tuple[head, dep, id_relation]]
        """

        try:
            example_enc = deepcopy(example)

            # tokens
            if add_bounds:
                example_enc.tokens = ["[START]"] + example_enc.tokens + ["[END]"]

            # labels
            labels_encoded = []
            for label in example.labels:
                label_enc = vocab_ner.get_id(label)
                labels_encoded.append(label_enc)
            if add_bounds:
                label = vocab_ner.get_id(cls.NER_LABEL_OTHER)
                labels_encoded = [label] + labels_encoded + [label]
            example_enc.labels = labels_encoded

            # arcs
            id2index = {x.id: i for i, x in enumerate(sorted(example.entities, key=lambda x: x.start))}
            arcs_encoded = []
            for arc in example.arcs:
                head = id2index[arc.head]
                dep = id2index[arc.dep]
                rel = vocab_re.get_id(arc.rel)
                arcs_encoded.append((head, dep, rel))
            example_enc.arcs = arcs_encoded
            return example_enc

        except Exception as e:
            print(e)
            print(f"strange example: {example.filename}")

    @staticmethod
    def check_example(example: Example, ner_encoding: str):
        """
        NER:
        * число токенов равно числу лейблов
        * entity.start >= entity.end
        * начало сущности >= 0, конец сущности < len(tokens)
        RE:
        * оба аргумента отношений есть в entities
        """
        assert ner_encoding in {"bio", "bilou"}

        assert len(example.tokens) == len(example.labels), \
            f"[{example.filename}] tokens and labels mismatch, {len(example.tokens)} != {len(example.labels)}"

        entity_ids = set()
        for entity in example.entities:
            assert entity.start <= entity.end, \
                f"[{example.filename}] strange entity span, start = {entity.start}, end = {entity.end}"
            assert entity.start >= 0, f"[{example.filename}] strange entity start: {entity.start}"
            assert entity.end < len(example.tokens), \
                f"[{example.filename}] strange entity end: {entity.end}, but num tokens is {len(example.tokens)}"
            entity_ids.add(entity.id)

        for arc in example.arcs:
            assert arc.head in entity_ids, \
                f"[{example.filename}] something is wrong with arc {arc.id}: head {arc.head} is unknown"
            assert arc.dep in entity_ids, \
                f"[{example.filename}] something is wrong with arc {arc.id}: dep {arc.dep} is unknown"

        arcs = [(arc.head, arc.dep, arc.rel) for arc in example.arcs]
        assert len(arcs) == len(set(arcs)), f"[{example.filename}] there duplicates in arcs"

        if ner_encoding == "bilou":
            num_start_ids = sum(x.startswith("B") for x in example.labels)
            num_end_ids = sum(x.startswith("L") for x in example.labels)
            assert num_start_ids == num_end_ids, \
                f"[{example.filename}]: num start ids: {num_start_ids}, num end ids: {num_end_ids}"

    @classmethod
    def fit_vocabs(cls, examples, ner_encoding: str):
        assert ner_encoding in {"bio", "bilou"}
        # labels vocab
        vocab_ner = set()
        prefixes = {"B", "I"}
        if ner_encoding == "bilou":
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
        vocab_ner.add(cls.NER_LABEL_OTHER)
        vocab_ner = Vocab(vocab_ner)

        # arcs vocab
        vocab_re = set()
        for x in examples:
            for arc in x.arcs:
                vocab_re.add(arc.rel)
        vocab_re.add(cls.RE_LABEL_OTHER)
        vocab_re = Vocab(vocab_re)

        return vocab_ner, vocab_re

    def _parse_example(self, filename: str, ner_encoding: str):
        """
        строчка файла filename:
        сущность:
        T5\tBIN 325 337\tФормирование\n
        отношение:
        R105\tTSK Arg1:T370 Arg2:T371
        """
        # подгрузка текста
        with open(os.path.join(self.data_dir, f'{filename}.txt')) as f:
            text = ' '.join(f)
            text = text.replace('\n ', '\n')

        # .ann
        span2label = {}  # span -> {"id": str, label: str}
        arcs = []
        arcs_used = set()  # в арках бывают дубликаты, пример: R35, R36 в 31339011023601075299026_18_part_1.ann
        with open(os.path.join(self.data_dir, f'{filename}.ann'), 'r') as f:
            for line in f:
                line = line.strip()
                content = line.split('\t')
                line_tag = content[0]
                if line_tag.startswith("T"):
                    try:
                        _, entity, expected_entity_pattern = content
                    except ValueError:
                        print(f"something is wrong with line: {line}")
                        return
                    label, start, end = entity.split()
                    start = int(start)
                    end = int(end)
                    actual_entity_pattern = text[start:end]
                    if actual_entity_pattern != expected_entity_pattern:
                        print(f"something is wrong with markup; expected entity is {expected_entity_pattern}, but "
                              f"got {actual_entity_pattern}")
                        return
                    entity_tokens = list(TOKENS_EXPRESSION.finditer(expected_entity_pattern))
                    num_entity_tokens = len(entity_tokens)
                    for i, m in enumerate(entity_tokens):
                        # вывод префикса:
                        if ner_encoding == "bio":
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
                        # создание лейбла
                        si, ei = m.span()
                        span = start + si, start + ei
                        ner_label = NERLabel(
                            id_entity=line_tag,
                            value=prefix + "_" + label
                        )
                        span2label[span] = ner_label
                elif line_tag.startswith("R"):
                    # TODO: может ли граф сущностей быть цикличным?
                    _, relation = content
                    label, arg1, arg2 = relation.split()
                    arc = Arc(
                        id=line_tag,
                        head=arg1.split(":")[1],
                        dep=arg2.split(":")[1],
                        rel=label
                    )
                    if (arc.head, arc.dep, arc.rel) not in arcs_used:
                        arcs.append(arc)
                        arcs_used.add((arc.head, arc.dep, arc.rel))
                else:
                    raise Exception(f"invalid line: {line}")

        # если не вылетела ошибка, то присвоим каждому токену текста лейбл
        tokens = []
        labels = []
        for i, m in enumerate(TOKENS_EXPRESSION.finditer(text)):
            span = m.span()
            label = span2label.get(span, NERLabel(id_entity=None, value=self.NER_LABEL_OTHER))
            tokens.append(m.group())
            labels.append(label)

        entities = self._get_entities_spans(labels)

        # больше айдишник сущности не нужен
        labels = [x.value for x in labels]

        example = Example(
            filename=filename,
            text=text,
            tokens=tokens,
            labels=labels,
            entities=entities,
            arcs=arcs
        )
        # self._fill_arcs(example)
        return example

    @classmethod
    def _get_entities_spans(cls, labels: List[NERLabel]):
        entities = []
        # prev_label = ""  # предыдущий таг
        prev_label = NERLabel(id_entity=None, value="")
        id_entity = None
        entity_token_ids = []

        def add_new_entity():
            if not entity_token_ids:
                return
            entity = Entity(
                id=id_entity,
                start=entity_token_ids[0],
                end=entity_token_ids[-1]
            )
            entities.append(entity)
            entity_token_ids.clear()

        for i, label in enumerate(labels):
            v = label.value
            tag = v.split("_")[-1]
            prev_tag = prev_label.value.split("_")[-1]
            if v != cls.NER_LABEL_OTHER:
                if v.startswith("B") or v.startswith("U"):  # начало новой сущности
                    add_new_entity()
                else:
                    if tag != prev_tag:  # начало новой сущности
                        add_new_entity()
                id_entity = label.id_entity
                entity_token_ids.append(i)
            else:
                add_new_entity()

            prev_label = label

        add_new_entity()

        return entities


class ParserRuREBusV2(ParserRuREBus):
    """
    1. при решении задачи ner -> re end2end спаны сущностей на уровне токенов нам неизвестны.
    следовательно, они должы выводиться на уровне вычислительного графа из лейблов токенов.
    2. при разделении документа на предложения учитываются спаны сущностей, чтоб не допускать того,
    что одна часть сущности в одном кусочке, а другая - в другом.
    """
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def split_example(example: Example, window: int = 1):
        """
        Если example построен на уровне документа, то нужно разбить его
        на куски размера window предложений.
        """
        sentences = ru_sent_tokenize(example.text)
        lengths = [len(s) for s in sentences]
        assert sum(lengths) == len(example.tokens)

        examples = []

        # Entity -> Set[Arc]
        id_entity2arcs = defaultdict(set)

        for arc in example.arcs:
            id_entity2arcs[arc.head].add(arc)
            id_entity2arcs[arc.dep].add(arc)

        for id_sent in range(0, len(lengths) - window + 1):
            start = sum(lengths[:id_sent])
            end = start + sum(lengths[id_sent:id_sent + window])

            tokens_new = example.tokens[start:end]
            labels_new = example.labels[start:end]

            # нужно взять только те сущности entity, что:
            # * start <= entity.start <= end
            # * start <= entity.end <= end
            entities_new = []
            arcs_new = {}
            for i_old, entity in enumerate(example.entities):
                # [      (            )          ]
                # start  start_entity end_entity end
                if start <= entity.start and entity.end < end:
                    start_entity_new = entity.start - start
                    end_entity_new = entity.end - start
                    entity_new = Entity(id=entity.id, start=start_entity_new, end=end_entity_new)
                    entities_new.append(entity_new)

                    for arc in id_entity2arcs[entity.id]:
                        arcs_new[arc.id] = deepcopy(arc)

            arcs_new = list(arcs_new.values())

            text_new = ' '.join(sentences[id_sent:id_sent + window])
            x_new = Example(
                filename=example.filename,
                text=text_new,
                tokens=tokens_new,
                labels=labels_new,
                entities=entities_new,
                arcs=arcs_new,
            )
            examples.append(x_new)

        return examples

    @classmethod
    def encode_example(cls, example: Example, vocab_ner: Vocab, vocab_re: Vocab, add_bounds: bool = False):
        """
        Кодирование категориальных атрибутов примеров:
        * tokens - List[str] (остаётся неизменным)
        * labels - List[int]
        * arcs - List[Tuple[head, dep, id_relation]]
        """

        try:
            example_enc = deepcopy(example)

            # tokens
            if add_bounds:
                example_enc.tokens = ["[START]"] + example_enc.tokens + ["[END]"]

            # labels
            labels_encoded = []
            for label in example.labels:
                label_enc = vocab_ner.get_id(label)
                labels_encoded.append(label_enc)
            if add_bounds:
                label = vocab_ner.get_id(cls.NER_LABEL_OTHER)
                labels_encoded = [label] + labels_encoded + [label]
            example_enc.labels = labels_encoded

            # arcs
            id2index = {x.id: i for i, x in enumerate(sorted(example.entities, key=lambda x: x.start))}
            arcs_encoded = []
            for arc in example.arcs:
                head = id2index[arc.head]
                dep = id2index[arc.dep]
                rel = vocab_re.get_id(arc.rel)
                arcs_encoded.append((head, dep, rel))
            example_enc.arcs = arcs_encoded
            return example_enc

        except Exception as e:
            print(e)
            print(f"strange example: {example.filename}")

    def _parse_example(self, filename: str, ner_encoding: str):
        """
        строчка файла filename:
        сущность:
        T5\tBIN 325 337\tФормирование\n
        отношение:
        R105\tTSK Arg1:T370 Arg2:T371
        """
        # подгрузка текста
        with open(os.path.join(self.data_dir, f'{filename}.txt')) as f:
            text = ' '.join(f)
            text = text.replace('\n ', '\n')

        # .ann
        span2label = {}  # span -> {"id": str, label: str}
        arcs = []
        arcs_used = set()  # в арках бывают дубликаты, пример: R35, R36 в 31339011023601075299026_18_part_1.ann
        with open(os.path.join(self.data_dir, f'{filename}.ann'), 'r') as f:
            for line in f:
                line = line.strip()
                content = line.split('\t')
                line_tag = content[0]
                if line_tag.startswith("T"):
                    try:
                        _, entity, expected_entity_pattern = content
                    except ValueError:
                        print(f"something is wrong with line: {line}")
                        return
                    label, start, end = entity.split()
                    start = int(start)
                    end = int(end)
                    actual_entity_pattern = text[start:end]
                    if actual_entity_pattern != expected_entity_pattern:
                        print(f"something is wrong with markup; expected entity is {expected_entity_pattern}, but "
                              f"got {actual_entity_pattern}")
                        return
                    entity_tokens = list(TOKENS_EXPRESSION.finditer(expected_entity_pattern))
                    num_entity_tokens = len(entity_tokens)
                    for i, m in enumerate(entity_tokens):
                        # вывод префикса:
                        if ner_encoding == "bio":
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
                        # создание лейбла
                        si, ei = m.span()
                        span = start + si, start + ei
                        ner_label = NERLabel(
                            id_entity=line_tag,
                            value=prefix + "_" + label
                        )
                        span2label[span] = ner_label
                elif line_tag.startswith("R"):
                    _, relation = content
                    label, arg1, arg2 = relation.split()
                    arc = Arc(
                        id=line_tag,
                        head=arg1.split(":")[1],
                        dep=arg2.split(":")[1],
                        rel=label
                    )
                    if (arc.head, arc.dep, arc.rel) not in arcs_used:
                        arcs.append(arc)
                        arcs_used.add((arc.head, arc.dep, arc.rel))
                else:
                    raise Exception(f"invalid line: {line}")

        # если не вылетела ошибка, то присвоим каждому токену текста лейбл
        tokens = []
        labels = []
        matches = []
        for i, m in enumerate(TOKENS_EXPRESSION.finditer(text)):
            span = m.span()
            label = span2label.get(span, NERLabel(id_entity=None, value=self.NER_LABEL_OTHER))
            tokens.append(m.group())
            labels.append(label)
            matches.append(m)

        entities = self._get_entities_spans_v2(matches, labels)

        # больше айдишник сущности не нужен
        labels = [x.value for x in labels]

        example = Example(
            filename=filename,
            text=text,
            tokens=tokens,
            labels=labels,
            entities=entities,
            arcs=arcs
        )
        # self._fill_arcs(example)
        return example

    @classmethod
    def _get_entities_spans_v2(cls, matches: List, labels: List[NERLabel]):
        entities = []
        # prev_label = ""  # предыдущий таг
        prev_label = NERLabel(id_entity=None, value="")
        id_entity = None
        entity_matches = []

        def add_new_entity():
            if not entity_matches:
                return
            entity = Entity(
                id=id_entity,
                start=entity_matches[0].start(),
                end=entity_matches[-1].end()  # нужен для разделения на предложения
            )
            entities.append(entity)
            entity_matches.clear()

        for m, label in zip(matches, labels):
            v = label.value
            tag = v.split("_")[-1]
            prev_tag = prev_label.value.split("_")[-1]
            if v != cls.NER_LABEL_OTHER:
                if v.startswith("B") or v.startswith("U") or (tag != prev_tag):  # начало новой сущности
                    add_new_entity()
                id_entity = label.id_entity
                entity_matches.append(m)
            else:
                add_new_entity()

            prev_label = label

        add_new_entity()

        return entities
