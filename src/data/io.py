import os
import shutil
import re
import tqdm
import unicodedata
from typing import List, Union, Pattern, Callable, IO
from collections import defaultdict

from src.data.base import (
    Attribute,
    Arc,
    Entity,
    Event,
    EventArgument,
    Example,
    LineTypes,
    NerEncodings,
    NerPrefixJoiners,
    Span,
    Token,
    TOKENS_EXPRESSION
)
from src.data.exceptions import (
    BadLineError,
    EntitySpanError,
    MultiRelationError,
    NestedNerError,
    NestedNerSingleEntityTypeError,
    RegexError
)


def parse_collection(
        data_dir: str,
        n: int = None,
        tokens_pattern: Union[str, Pattern] = None,
        ignore_bad_examples: bool = False,
        read_fn: Callable = None
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
    if tokens_pattern is not None:
        if isinstance(tokens_pattern, str):
            tokens_expression = re.compile(tokens_pattern)
        else:
            tokens_expression = tokens_pattern
    else:
        tokens_expression = TOKENS_EXPRESSION

    # парсим примеры для обучения
    examples = []
    error_counts = defaultdict(int)
    for filename in tqdm.tqdm(names_to_parse):
        try:
            example = parse_example(
                data_dir=data_dir,
                filename=filename,
                tokens_expression=tokens_expression,
                read_fn=read_fn
            )
            examples.append(example)
        except (BadLineError, MultiRelationError, NestedNerError, NestedNerSingleEntityTypeError, RegexError) as e:
            err_name = type(e).__name__
            print(f"[{filename}] known error {err_name} occurred:")
            print(e)
            if ignore_bad_examples:
                print("example ignored due to flag ignore_bad_examples set to True")
                print("=" * 50)
                error_counts[err_name] += 1
            else:
                raise e
        except EntitySpanError as e:
            err_name = type(e).__name__
            print(f"[{filename}] known error {err_name} occurred:")
            print(e)
            print("trying another readers...")
            flag = False
            for read_fn_alt in [read_file_v1, read_file_v2, read_file_v3]:
                print("reader:", read_fn_alt.__name__)
                if read_fn_alt.__name__ == read_fn.__name__:
                    print("ignored due to the same as provided in args")
                    continue
                try:
                    example = parse_example(
                        data_dir=data_dir,
                        filename=filename,
                        tokens_expression=tokens_expression,
                        read_fn=read_fn_alt
                    )
                    examples.append(example)
                    flag = True
                    break
                except EntitySpanError as e:
                    print(e)
            if flag:
                print("success :)")
            else:
                print("fail :(")

        except Exception as e:
            print(f"[{filename}] unknown error {type(e).__name__} occurred:")
            raise e
    print(f"successfully parsed {len(examples)} examples from {len(names_to_parse)} files.")
    print(f"error counts: {error_counts}")
    return examples


def read_file_v1(f: IO) -> str:
    text = f.read()
    return text


def read_file_v2(f: IO) -> str:
    """
    collection5
    """
    text = " ".join(f)
    return text


def read_file_v3(f: IO) -> str:
    """
    rured, rucor, rurebus
    """
    text = " ".join(f)
    text = text.replace('\n ', '\n')
    return text


def parse_example(
        data_dir: str,
        filename: str,
        tokens_expression: Pattern = TOKENS_EXPRESSION,
        read_fn: Callable = None
) -> Example:
    """
    строчка файла filename.ann:

    * сущность: T5\tBIN 325 337\tФормирование\n
    * отношение: R105\tTSK Arg1:T370 Arg2:T371\n
    * событие: E0\tBankruptcy:T0 Bankrupt:T1 Bankrupt2:T2\n
    * атрибут: A1\tEvent-time E0 Past\n
    * комментарий: #0\tAnnotatorNotes T3\tfoobar\n

    замечения по файлам .ann:
    * факт наличия нескольких событий может быть представлен несколькими способыми:
    Компания ООО "Ромашка" обанкротилась
    T0  Bankruptcy 1866 1877    банкротства
    E0  Bankruptcy:T0
    T1  AnotherEvent 1866 1877    банкротства
    E1  AnotherEvent:T1

    T0  Bankruptcy 1866 1877    банкротства
    E0  Bankruptcy:T0
    E1  AnotherEvent:T0

    E0  Bankruptcy:T0
    E1  AnotherEvent:T0
    T0  Bankruptcy 1866 1877    банкротства

    * аргумент атрибута или комментария всегда указан раньше, чем сам атрибут.
    * если комментируется события, то аргументом комментария является идентификатор события,
    а не идекнтификатор триггера.
    T12     Bankruptcy 1866 1877    банкротства
    E3      Bankruptcy:T12
    A10     Negation E3
    #1      AnnotatorNotes E3  foobar
    """
    # подгрузка текста
    read_fn = read_fn if read_fn is not None else read_file_v1
    with open(os.path.join(data_dir, f'{filename}.txt')) as f:
        text = read_fn(f)

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
                # TODO: делать это вне этой функции
                # entity_label = fix_entity_label(label=entity_label, ner_prefix_joiner=ner_prefix_joiner)
                entity_span = start_index, end_index

                # проверка того, что в файле .txt в спане из файла .ann находится
                # правильная именная сущность
                actual_entity_pattern = text[start_index:end_index]
                if actual_entity_pattern != expected_entity_pattern:
                    raise EntitySpanError(f"[{filename}]: something is wrong with markup; "
                                          f"expected entity is <bos>{expected_entity_pattern}<eos>, "
                                          f"but got <bos>{actual_entity_pattern}<eos>")

                # создание сущности
                entity = Entity(
                    id=line_tag,
                    label=entity_label,
                    text=actual_entity_pattern,
                    tokens=None,
                    is_event_trigger=False  # заполнится потом
                )
                entity.span = entity_span  # TODO: временное решение
                id2entity[entity.id] = entity
                id2arg[entity.id] = entity

            # отношение
            elif line_type == LineTypes.RELATION:
                try:
                    _, relation = content
                    re_label, arg1, arg2 = relation.split()
                except ValueError:
                    raise BadLineError(f"[{filename}]: something is wrong with line: {line}")
                head = arg1.split(":")[1]
                dep = arg2.split(":")[1]
                arc = Arc(id=line_tag, head=head, dep=dep, rel=re_label)
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
                    rel = remove_role_index(rel)

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

    arcs = list(id2arc.values())
    events = list(id2event.values())

    # сущности: расставление флагов is_event_trigger
    # оказывается, событие может быть указано раньше триггера в файле .ann
    for event in events:
        id2entity[event.trigger].is_event_trigger = True

    entities = list(id2entity.values())

    # очистка текста от "плохих" символов
    bad_ids = get_invalid_char_indices(text)
    text_clean = remove_bad_ids(text, bad_ids)

    # токенизация текста
    tokens = []
    start2token = {}
    for i, m in enumerate(tokens_expression.finditer(text_clean)):
        span = Span(*m.span())
        token = Token(
            text=m.group(),
            span_abs=span,
            span_rel=span,
            index_abs=i,
            index_rel=i,
        )
        tokens.append(token)
        start2token[span.start] = token

    # очистка названия сущности от "плохих символов"
    # фикс спана сущности
    # присвоение токенов сущности
    for entity in entities:
        start, end = entity.span
        start_new = start
        end_new = end
        bad_ids_entity = []
        for i in bad_ids:
            if i <= start:
                start_new -= 1
                end_new -= 1
                if i == start:
                    bad_ids_entity.append(0)
            elif start < i < end:
                end_new -= 1
                bad_ids_entity.append(i - start)
        entity.span = start_new, end_new
        entity.text = remove_bad_ids(entity.text, bad_ids_entity)
        if text_clean[start_new:end_new] != entity.text:
            raise RegexError(f"[{filename}]  {text_clean[start_new:end_new]} != {entity.text}. "
                             f"Check entity {entity.id} in {filename}.ann file")

        entity_tokens = []
        for i in range(start_new, end_new):
            if i in start2token:
                entity_tokens.append(start2token[i])
        entity.tokens = entity_tokens

    # создание инстанса класса Example
    example = Example(
        filename=filename,
        id=filename,
        text=text_clean,
        tokens=tokens,
        entities=entities,
        arcs=arcs,
        events=events
    )

    return example


# TODO: докинуть сюда логику с вложенным нером
def is_valid_example(
        x: Example,
        allow_nested_entities: bool = False,
        allow_nested_entities_one_label: bool = False,
        allow_many_entities_per_span_one_label: bool = False,
        allow_many_entities_per_span_different_labels: bool = False
):
    """

    :param x:
    :param allow_nested_entities: разрешена ли вложенность такого вида: <ORG> foo <LOC> bar </LOC></ORG>
    :param allow_nested_entities_one_label: разрешена ли вложенность такого вида: <ORG> foo <ORG> bar </ORG></ORG>
    :param allow_many_entities_per_span_one_label: разрешена ли вложенность такого вида: <ORG> foo <ORG> bar </ORG></ORG>
    :param allow_many_entities_per_span_different_labels: разрешена ли вложенность такого вида: <ORG><LOC>foo</LOC></ORG>
    :return:
    """
    # проверка того, что в файле .ann нет дубликатов по сущностям
    span2label = {}
    for entity in x.entities:
        if entity.span in span2label:
            if span2label[entity.span] == entity.label:
                if not allow_many_entities_per_span_one_label:
                    raise EntitySpanError(f"[{x.filename}]: tried to assign one more label {entity.label} "
                                          f"to span {entity.span}")
            else:
                if not allow_many_entities_per_span_different_labels:
                    raise EntitySpanError(f"[{x.filename}]: span {entity.span} has already "
                                          f"label {span2label[entity.span]},"
                                          f"but tried to assign also label {entity.label}")
        else:
            span2label[entity.span] = entity.label


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def get_invalid_char_indices(text):
    res = []
    for i in range(len(text)):
        char = text[i]
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            res.append(i)
    return res


def remove_bad_ids(s, bad_ids):
    s_clean = ""
    start = 0
    for end in bad_ids:
        s_clean += s[start:end]
        start = end + 1
    if start < len(s):
        s_clean += s[start:]
    return s_clean


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


def fix_entity_label(label: str, ner_prefix_joiner: str) -> str:
    """
    Нужна гарантия того, что лейбл не содержит разделитель
    """
    if ner_prefix_joiner == NerPrefixJoiners.HYPHEN:
        repl = NerPrefixJoiners.UNDERSCORE
    elif ner_prefix_joiner == NerPrefixJoiners.UNDERSCORE:
        repl = NerPrefixJoiners.HYPHEN
    else:
        raise Exception
    label = label.replace(ner_prefix_joiner, repl)
    return label


def remove_role_index(s: str) -> str:
    """
    если с сущностью E связано несколько сущностей отношением R,
    то к отношению добавляется индекс.
    Пример: [ORG1 ООО "Ромашка"] и [ORG2 ПАО "Одуванчик"] признаны [BANKRUPTCY банкротами]
    в файле .ann это будет записано так:
    T1 ORG start1 end1 ООО "Ромашка"
    T2 ORG start2 end2 ПАО "Одуванчик"
    T3 BANKRUPTCY start3 end3 банкротами
    E0 BANKRUPTCY:T3 Bankrupt1:T1 Bankrupt2:T2
    чтобы понять, что Bankrupt1 и Bankrupt2 - одна и та же роль, нужно убрать индекс в конце
    """
    matches = list(re.finditer(r"\d+", s))
    if matches:
        m = matches[-1]
        start, end = m.span()
        if end == len(s):
            s = s[:start]
    return s


def simplify(example: Example):
    """
    упрощение графа путём удаления тривиальных сущностей и рёбер

    нужно игнорить:
    * дубликаты триггеров:
    T0  Bankruptcy 1866 1877    банкротства
    T1  Bankruptcy 1866 1877    банкротства
    * дубликаты событий:
    E0  Bankruptcy:T0
    E1  Bankruptcy:T0
    * дубликаты рёбер:
    E0  Bankruptcy:T0 EventArg:T2
    E1  Bankruptcy:T0 EventArg:T2

    сущность: одному спану соответствует не более одной сущности
    событие: одной сущности соответствует не более одного события
    ребро: одной паре спанов соответствует не более одного ребра

    предпосылка: одному спану соответствует один лейбл

    :param example:
    :return:
    """
    assert isinstance(example.id, str)

    span_to_entities = defaultdict(set)
    span_pair_to_arcs = defaultdict(set)

    # событие: {id, trigger, label}
    # @ trigger.is_event_flag = True
    # @ label == trigger.label
    # то есть по сути инстансы класса Event избыточны

    id2entity = {}
    for entity in example.entities:
        id2entity[entity.id] = entity
        span = entity.tokens[0].span_abs.start, entity.tokens[-1].span_abs.end
        span_to_entities[span].add(entity)

    id2event = {event.id: event for event in example.events}
    for arc in example.arcs:
        # arc.head и arc.dep могут быть T и E
        if arc.head[0] == LineTypes.ENTITY:
            head = id2entity[arc.head]
        elif arc.head[0] == LineTypes.EVENT:
            event = id2event[arc.head]
            head = id2entity[event.trigger]
        else:
            raise

        if arc.dep[0] == LineTypes.ENTITY:
            dep = id2entity[arc.dep]
        elif arc.dep[0] == LineTypes.EVENT:
            event = id2event[arc.dep]
            dep = id2entity[event.trigger]
        else:
            raise

        key = (head.tokens[0].span_abs.start, head.tokens[-1].span_abs.end), \
              (dep.tokens[0].span_abs.start, dep.tokens[-1].span_abs.end)
        span_pair_to_arcs[key].add(arc)

    entities_new = []
    span_to_id = {}
    id_span = 0
    for span, entities in span_to_entities.items():
        unique_labels = {x.label for x in entities}
        assert len(unique_labels) == 1, \
            f"[{example.id}] expected one unique label per span, but got {unique_labels} for span {span}"
        entity = entities.pop()
        _id = f"T{id_span}"
        entity_new = Entity(
            id=_id,
            label=entity.label,
            text=entity.text,
            tokens=entity.tokens,
            is_event_trigger=entity.is_event_trigger,
            attrs=entity.attrs.copy(),
            comment=entity.comment
        )
        entities_new.append(entity_new)
        span_to_id[span] = _id
        id_span += 1

    arcs_new = []
    id_span_pair = 0
    for (span_head, span_dep), arcs in span_pair_to_arcs.items():
        unique_labels = {x.rel for x in arcs}
        assert len(unique_labels) == 1, f"[{example.id}] expected one unique label per span pair, " \
            f"but got {unique_labels} for span pair ({span_head}, {span_dep})"
        arc = arcs.pop()
        arc_new = Arc(
            id=f"R{id_span_pair}",
            head=span_to_id[span_head],
            dep=span_to_id[span_dep],
            rel=arc.rel
        )
        arcs_new.append(arc_new)
        id_span_pair += 1

    example_copy = Example(
        filename=example.filename,
        id=example.id,
        text=example.text,
        tokens=example.tokens,
        entities=entities_new,
        arcs=arcs_new,
        label=example.label
    )

    return example_copy


def to_conll(examples, path):
    """
    формат:
    #begin document <doc_name_1>
    token_0\tlabel_0\n
    ...
    token_k\tlabel_k\n
    #end document

    #begin document <doc_name_2>
    ...

    token - токен
    label - выражение, по которому можно понять, к каким сущностям и компонентам принадлежит токен (см. примеры ниже)

    * в одном файле может быть несколько документов
    * примеры разметки в случае вложенных сущностей:

    ```
    Члены   (235
    Талибана        (206)|235)
    сейчас  -
    находится       -
    в       -
    бегах   -
    ```
    сущность "Талибана" принадлежит к компоненте связности 206
    и вложена в сущность "Члены Талибана", которая принадлежит компоненете связности 235

    ```
    первые  -
    полученные      -
    деньги  -
    должны  -
    быть    -
    потрачены       -
    на      -
    восстановление  -
    всех    (43
    разрушенных     -
    бомбежкой       -
    школ    -
    для     -
    девочек -
    в       -
    Свате   43)|(50)
    .       -
    ```
    сущность "Свате" принадлежит к компоненте связности 50
    и вложена в сущность "всех разрушенных бомбежкой школ для девочек в Свате",
    которая принадлежит компоненете связности 43

    при закрытии вложенности не обязательно указывать номера компонент в порядке, соответствующем порядку открытия.
    можно заметить, что в первом примере сначала указана внутренняя сущность (206), а потом закрыта внешняя (235).
    Во втором примере ситуация обратная. В обоих случаях ошибки не будет.

    После генерации файлов в conll-формате оценка качества производится запуском скрипта scorer.pl
    из библиотеки https://github.com/conll/reference-coreference-scorers:

    perl scorer.pl <metric> <key> <response> [<document-id>]
    <metric> - название метрики (all, если интересуют все)
    <key> - y_true
    <response> - y_pred
    <document-id> (optional) - номер документа, на котором хочется померить качество
    (none, если интересует качество на всём корпусе)
    подробное описание см. в README библиотеки
    """
    # не множество, так как могут быть дубликаты: например, если две сущности
    # начинаются с разных токенов, но заканчиваются в одном, причём относятся к одной копмоненте
    token2info = defaultdict(list)
    for x in examples:
        for entity in x.entities:
            assert entity.id_chain is not None
            index_start = entity.tokens[0].index_abs
            index_end = entity.tokens[-1].index_abs
            _is_single = index_start == index_end
            token2info[(x.id, index_start)].append((entity.id_chain, _is_single, True))
            token2info[(x.id, index_end)].append((entity.id_chain, _is_single, False))

    def build_label(id_example, token_index) -> str:
        """
        token.index_abs -> множество компонент которые он {открывает, закрывает}
        если ничего не открывает и не закрывает, то вернуть "-"
        """
        key = id_example, token_index
        if key in token2info:
            items = token2info[key]
            pieces = []
            singles = set()
            for id_chain, is_single, is_open in items:
                if is_single:
                    if id_chain in singles:
                        continue
                    else:
                        p = f'({id_chain})'
                        singles.add(id_chain)
                else:
                    if is_open:
                        p = f'({id_chain}'
                    else:
                        p = f'{id_chain})'
                pieces.append(p)
            res = '|'.join(pieces)
        else:
            res = "-"
        return res

    with open(path, "w") as f:
        for x in examples:
            num_open = 0
            num_close = 0
            f.write(f"#begin document {x.id}\n")
            for t in x.tokens:
                label = build_label(id_example=x.id, token_index=t.index_abs)
                num_open += label.count('(')
                num_close += label.count(')')
                f.write(f"{t.text}\t{label}\n")
            f.write("#end document\n\n")
            assert num_open == num_close, f"[{x.id}] {num_open} != {num_close}"


# TODO: создавать инстансы класса Event на уровне model.predict
def to_brat(
        examples: List[Example],
        output_dir: str,
        write_mode: str = "a"
):
    assert write_mode in {"a", "w"}
    os.makedirs(output_dir, exist_ok=True)
    event_counter = defaultdict(int)
    filenames = set()
    for x in examples:
        filenames.add(x.filename)

        with open(os.path.join(output_dir, f"{x.filename}.txt"), write_mode) as f:
            f.write(x.text)

        with open(os.path.join(output_dir, f"{x.filename}.ann"), write_mode) as f:
            events = {}
            # сущности
            for entity in x.entities:
                start = entity.tokens[0].span_abs.start
                end = entity.tokens[-1].span_abs.end
                assert isinstance(entity.id, str)
                assert entity.id[0] == "T"
                line = f"{entity.id}\t{entity.label} {start} {end}\t{entity.text}\n"
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
                assert isinstance(arc.rel, str), "forget to transform arc codes to values!"
                if arc.head in events:
                    arg = EventArgument(id=arc.dep, role=arc.rel)
                    events[arc.head].args.append(arg)
                else:
                    id_arc = get_id(arc.id, "R")
                    line = f"{id_arc}\t{arc.rel} Arg1:{arc.head} Arg2:{arc.dep}\n"
                    f.write(line)

            # события
            for event in events.values():
                assert event.id is not None
                id_event = get_id(event.id, "E")
                line = f"{id_event}\t{event.label}:{event.trigger}"
                role2count = defaultdict(int)
                args_str = ""
                for arg in event.args:
                    i = role2count[arg.role]
                    role = arg.role
                    if i > 0:
                        role += str(i + 1)
                    args_str += f"{role}:{arg.id}" + ' '
                    role2count[arg.role] += 1
                args_str = args_str.rstrip()
                if args_str:
                    line += ' ' + args_str
                line += '\n'
                f.write(line)


def to_brat_v2(examples: List[Example], output_dir: str,):
    """
    без триггеров событий
    :param examples:
    :param output_dir:
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    for x in examples:
        with open(os.path.join(output_dir, f"{x.filename}.txt"), "w") as f:
            f.write(x.text)

        with open(os.path.join(output_dir, f"{x.filename}.ann"), "w") as f:
            # сущности
            for entity in x.entities:
                start = entity.tokens[0].span_abs.start
                end = entity.tokens[-1].span_abs.end
                assert isinstance(entity.id, str)
                assert entity.id[0] == "T"
                line = f"{entity.id}\t{entity.label} {start} {end}\t{entity.text}\n"
                f.write(line)

            # отношения
            for arc in x.arcs:
                assert isinstance(arc.rel, str), "forget to transform arc codes to values!"
                id_arc = get_id(arc.id, "R")
                line = f"{id_arc}\t{arc.rel} Arg1:{arc.head} Arg2:{arc.dep}\n"
                f.write(line)


def get_id(id_arg: Union[int, str], prefix: str) -> str:
    assert id_arg is not None
    if isinstance(id_arg, str):
        assert len(id_arg) >= 2
        assert id_arg[0] == prefix
        assert id_arg[1:].isdigit()
        return id_arg
    elif isinstance(id_arg, int):
        return prefix + str(id_arg)
    else:
        raise ValueError(f"expected type of id_arg is string or integer, but got {type(id_arg)}")


# TODO: протестировать!!1!
def from_conllu(path: str, warn: bool = True) -> List[Example]:
    examples = []
    expression = re.compile(r'# sent_id = (.+\.xml)_(\d+)')
    num_chunks = 0
    num_chunks_ignored = 0
    num_tokens_total = 0
    num_tokens_chunk = 0
    chunks_i = []
    tokens_ij = []
    flag_strange = False
    filename_doc = None
    id_sent = None
    text = None

    def remove_spaces(s):
        return s.replace(' ', '').replace('\xa0', '')

    def append_chunk():
        id_chunk = f"{filename_doc}_{id_sent}"
        actual = ''.join(remove_spaces(_t.text) for _t in tokens_ij)
        expected = remove_spaces(text)
        if actual != expected:
            print(id_chunk)
            print("actual:", actual)
            print("expected:", expected)
            print("text:", text)
            print("tokens:", [_t.text for _t in tokens_ij])
            raise AssertionError
        chunk = Example(
            filename=filename_doc,
            id=id_chunk,
            text=text,
            tokens=tokens_ij.copy()
        )
        chunks_i.append(chunk)
        tokens_ij.clear()

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) == 0:
                continue

            if line[0] == "#":
                if line.startswith("# sent_id"):
                    if flag_strange:
                        flag_strange = False
                        num_chunks_ignored += 1
                        tokens_ij.clear()
                        num_tokens_chunk = 0
                    else:
                        if filename_doc is not None:
                            if len(tokens_ij) > 0:
                                append_chunk()
                                num_chunks += 1
                                num_tokens_total += num_tokens_chunk
                                num_tokens_chunk = 0
                            else:
                                print(f"[{filename_doc}] chunk {id_sent} has no tokens")

                    m = expression.match(line)
                    filename_chunk = m.group(1)
                    id_sent = int(m.group(2))

                    if filename_doc is None:
                        filename_doc = filename_chunk
                    else:
                        if filename_doc != filename_chunk:
                            if len(chunks_i) > 0:
                                x = Example(filename=filename_doc, chunks=chunks_i.copy())
                                examples.append(x)
                                chunks_i.clear()
                            else:
                                print(f"[{filename_doc}] no valid chunks")
                            filename_doc = filename_chunk

                elif line.startswith("# text"):
                    text = line[9:]
            else:
                if flag_strange:
                    continue
                features = line.split("\t")
                token_id = features[0]
                token = features[1]
                pos = features[3]
                head = features[6]
                rel = features[7]

                try:
                    int(token_id)
                except ValueError:
                    if warn:
                        print(f"[{filename_doc}] [{id_sent}] strange token index {token_id}")
                    flag_strange = True

                try:
                    head = int(head)
                except ValueError:
                    if warn:
                        print(f"[{filename_doc}] [{id_sent}] strange head index {head} "
                              f"for token {token_id} <bos>{token}<eos>")
                    flag_strange = True

                if flag_strange:
                    continue

                if head == 0:
                    head = -1
                else:
                    head -= 1

                t = Token(
                    text=token,
                    id_head=head,
                    rel=rel,
                    pos=pos
                )
                tokens_ij.append(t)
                num_tokens_chunk += 1

    if flag_strange:
        num_chunks_ignored += 1
    else:
        if filename_doc is not None:
            if len(tokens_ij) > 0:
                append_chunk()
                num_chunks += 1
                num_tokens_total += num_tokens_chunk
            else:
                print(f"[{filename_doc}] chunk {id_sent} has no tokens")

    if len(chunks_i) > 0:
        x = Example(filename=filename_doc, chunks=chunks_i.copy())
        examples.append(x)
    else:
        print(f"[{filename_doc}] no valid chunks")

    n = sum(len(x.chunks) for x in examples)
    assert num_chunks == n, f"{num_chunks} != {n}"
    n = sum(sum(len(chunk.tokens) for chunk in x.chunks) for x in examples)
    assert num_tokens_total == n, f"{num_tokens_total} != {n}"

    print("===== DATASET INFO =====")
    print("num documents:", len(examples))
    print("num sentences:", num_chunks)
    print("num tokens:", num_tokens_total)
    print("num sentences ignored:", num_chunks_ignored)

    return examples


# if __name__ == "__main__":
#     path = "/home/vitaly/Desktop/ru_syntagrus-ud-dev.conllu"
#     from_conllu(path, warn=False)
#     data_dir = "/home/vitaly/brat-v1.3_Crunchy_Frog/data/examples/rured_fixed"
    # filename = "283"
    # x = parse_example_v2(data_dir=data_dir, filename=filename)
    # for file in os.listdir(data_dir):
    #     if file.endswith(".ann"):
    #         name = file.split(".")[0]
    #         try:
    #             parse_example_v2(data_dir=data_dir, filename=name)
    #         except BadLineError as e:
    #             print(e)
