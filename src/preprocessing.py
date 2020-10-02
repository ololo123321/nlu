import os
import re
import tqdm
from copy import deepcopy
from itertools import accumulate
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


class Example(ReprMixin):
    def __init__(self, filename=None, text=None, tokens=None, labels=None, entities=None, arcs=None):
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

    @property
    def chunks(self):
        """
        фикс. разделения на предложения: если начало сущности находится в одном предложении,
        а конец - в другом, то что-то не так с разделением на предложения, и соседние кусочки нужно склеить.
        по итогу должно гарантироваться, что одной сущности соответствует ровно одно предложение.
        """
        sent_candidates = ru_sent_tokenize(self.text)
        lengths = [len(TOKENS_EXPRESSION.findall(sent)) for sent in sent_candidates]
        assert sum(lengths) == len(self.tokens)

        sent_starts = [0] + list(accumulate(lengths))

        filename = self.filename  # чтоб self не прокидывать в аргумент ф-ии add_example
        sent_cum = ""
        entities_sorted = sorted(self.entities, key=lambda x: x.start_token_id)
        entities_cum = []
        tokens_cum = []
        labels_cum = []
        res = []

        def add_example():
            example_new = Example(
                filename=filename,
                text=sent_cum,
                tokens=tokens_cum.copy(),
                labels=labels_cum.copy(),
                entities=entities_cum.copy(),
                arcs=None
            )
            res.append(example_new)
            tokens_cum.clear()
            labels_cum.clear()
            entities_cum.clear()

        for i in range(len(sent_candidates)):
            sent_curr = sent_candidates[i]
            start = sent_starts[i]
            end = sent_starts[i + 1]
            tokens_cum += self.tokens[start:end]
            labels_cum += self.labels[start:end]
            # print("i:", i)
            # print("curr sent:", curr_sent)
            # print("cum sent:", cum_sent)
            if i == 0:
                sent_cum += sent_curr
                continue
            flag = True
            for entity in entities_sorted:
                print(start, end, entity)

                # вся сущность в предложении
                if start <= entity.start_token_id <= entity.end_token_id < end:
                    print(entity)
                    entities_cum.append(entity)

                # нет смысла бежать по сущностям, которые находятся дальше по тексту
                if entity.start_token_id >= end:
                    break

                # одна часть сущности в одном предложении, другая - в другом
                if entity.start_token_id < start <= entity.end_token_id:
                    print("entity start:", entity.start_token_id)
                    print("entity end:", entity.end_token_id)
                    print("sent start:", start)
                    print("id sent:", i)
                    sent_cum += ' ' + sent_curr
                    entities_cum.append(entity)
                    flag = False
                    break
            if flag:
                add_example()
                sent_cum = sent_curr

        add_example()

        # print("init num sentences:", len(sent_candidates))
        # print("new num sentences:", len(res))
        return res

    # @property
    # def chunks(self):
    #     """
    #     разделение исходного примера на несколько по предложениям.
    #     отношения такие, что одна сущность содержится в одном предложении, а другая - в другом, удаляются.
    #     """
    #     sentences = self.sentences_fixed
    #     lengths = [len(TOKENS_EXPRESSION.findall(sent)) for sent in sentences]
    #     assert sum(lengths) == len(self.tokens)
    #
    #     sent_starts = [0] + list(accumulate(lengths))
    #
    #     res = []
    #     entity_ptr = 0
    #     for i in range(len(sentences)):
    #         start = sent_starts[i]
    #         end = sent_starts[i + 1]
    #
    #         entities_i = []
    #         for entity in self.entities[entity_ptr:]:
    #             if entity.start_token_id > entity_ptr
    #             if start <= entity.start_token_id <= entity.end_token_id < end:
    #                 entities_i.append(entity)
    #
    #         example = Example(
    #             filename=self.filename,
    #             text=sentences[i],
    #             tokens=self.tokens[start:end],
    #             labels=self.labels[start:end],
    #             entities=None,
    #             arcs=None
    #         )
    #         res.append(example)
    #     return res


class ParserRuREBus:
    """
    https://github.com/dialogue-evaluation/RuREBus
    """
    NER_LABEL_OTHER = '0'
    RE_LABEL_OTHER = 'O'

    def __init__(self, ner_encoding):
        assert ner_encoding in {"bio", "bilou"}
        self.ner_encoding = ner_encoding

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

    def check_example(self, example: Example):
        """
        NER:
        * число токенов равно числу лейблов
        * entity.start >= entity.end
        * начало сущности >= 0, конец сущности < len(tokens)
        RE:
        * оба аргумента отношений есть в entities
        """
        assert len(example.tokens) == len(example.labels), \
            f"[{example.filename}] tokens and labels mismatch, {len(example.tokens)} != {len(example.labels)}"

        entity_ids = set()
        for entity in example.entities:
            assert entity.start_token_id <= entity.end_token_id, \
                f"[{example.filename}] strange entity span, start = {entity.start_token_id}, end = {entity.end_token_id}"
            assert entity.start_token_id >= 0, f"[{example.filename}] strange entity start: {entity.start_token_id}"
            assert entity.end_token_id < len(example.tokens), \
                f"[{example.filename}] strange entity end: {entity.end_token_id}, but num tokens is {len(example.tokens)}"
            entity_ids.add(entity.id)

        for arc in example.arcs:
            assert arc.head in entity_ids, \
                f"[{example.filename}] something is wrong with arc {arc.id}: head {arc.head} is unknown"
            assert arc.dep in entity_ids, \
                f"[{example.filename}] something is wrong with arc {arc.id}: dep {arc.dep} is unknown"

        arcs = [(arc.head, arc.dep, arc.rel) for arc in example.arcs]
        assert len(arcs) == len(set(arcs)), f"[{example.filename}] there duplicates in arcs"

        if self.ner_encoding == "bilou":
            num_start_ids = sum(x.startswith("B") for x in example.labels)
            num_end_ids = sum(x.startswith("L") for x in example.labels)
            assert num_start_ids == num_end_ids, \
                f"[{example.filename}]: num start ids: {num_start_ids}, num end ids: {num_end_ids}"

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

    def fit_vocabs(self, examples):
        # labels vocab
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
        vocab_ner.add(self.NER_LABEL_OTHER)
        vocab_ner = Vocab(vocab_ner)

        # arcs vocab
        vocab_re = set()
        for x in examples:
            for arc in x.arcs:
                vocab_re.add(arc.rel)
        vocab_re.add(self.RE_LABEL_OTHER)
        vocab_re = Vocab(vocab_re)

        return vocab_ner, vocab_re

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
        ner_labels = [self.NER_LABEL_OTHER] * len(text_tokens)
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
            filename=filename,
            text=text,
            tokens=text_tokens,
            labels=ner_labels,
            entities=entities,
            arcs=arcs
        )

        return example
