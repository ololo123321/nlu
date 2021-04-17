import os
import shutil
import re
import tqdm
from typing import List, Union, Pattern, Callable, IO
from collections import defaultdict

from .base import (
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
from .exceptions import (
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
        ner_encoding: str = NerEncodings.BIO,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        tokens_pattern: Union[str, Pattern] = None,
        allow_nested_entities: bool = False,
        allow_nested_entities_one_label: bool = False,
        allow_many_entities_per_span_one_label: bool = False,
        allow_many_entities_per_span_different_labels: bool = False,
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
                ner_encoding=ner_encoding,
                ner_prefix_joiner=ner_prefix_joiner,
                tokens_expression=tokens_expression,
                allow_nested_entities=allow_nested_entities,
                allow_nested_entities_one_label=allow_nested_entities_one_label,
                allow_many_entities_per_span_one_label=allow_many_entities_per_span_one_label,
                allow_many_entities_per_span_different_labels=allow_many_entities_per_span_different_labels,
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
                        ner_encoding=ner_encoding,
                        ner_prefix_joiner=ner_prefix_joiner,
                        tokens_expression=tokens_expression,
                        allow_nested_entities=allow_nested_entities,
                        allow_nested_entities_one_label=allow_nested_entities_one_label,
                        allow_many_entities_per_span_one_label=allow_many_entities_per_span_one_label,
                        allow_many_entities_per_span_different_labels=allow_many_entities_per_span_different_labels,
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
        ner_encoding: str = NerEncodings.BIO,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        tokens_expression: Pattern = TOKENS_EXPRESSION,
        allow_nested_entities: bool = False,
        allow_nested_entities_one_label: bool = False,
        allow_many_entities_per_span_one_label: bool = False,
        allow_many_entities_per_span_different_labels: bool = False,
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

    allow_nested_entities: разрешена ли вложенность такого вида: <ORG> foo <LOC> bar </LOC></ORG>
    allow_nested_entities_one_label: разрешена ли вложенность такого вида: <ORG> foo <ORG> bar </ORG></ORG>
    allow_many_entities_per_span_one_label: разрешена ли вложенность такого вида: <ORG><ORG>foo</ORG></ORG>
    allow_many_entities_per_span_different_labels: разрешена ли вложенность такого вида: <ORG><LOC>foo</LOC></ORG>
    """
    # подгрузка текста
    read_fn = read_fn if read_fn is not None else read_file_v1
    with open(os.path.join(data_dir, f'{filename}.txt')) as f:
        text = read_fn(f)

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
    for i, m in enumerate(tokens_expression.finditer(text)):
        span = Span(*m.span())
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

    span2label = {}

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
                entity_label = fix_entity_label(label=entity_label, ner_prefix_joiner=ner_prefix_joiner)

                # проверка того, что в файле .txt в спане из файла .ann находится
                # правильная именная сущность
                actual_entity_pattern = text[start_index:end_index]
                if actual_entity_pattern != expected_entity_pattern:
                    raise EntitySpanError(f"[{filename}]: something is wrong with markup; "
                                          f"expected entity is <bos>{expected_entity_pattern}<eos>, "
                                          f"but got <bos>{actual_entity_pattern}<eos>")

                # проверка на то, что в файле .ann нет дубликатов по сущностям
                entity_span = start_index, end_index
                if entity_span in span2label:
                    if span2label[entity_span] == entity_label:
                        if not allow_many_entities_per_span_one_label:
                            raise EntitySpanError(f"[{filename}]: tried to assign one more label {entity_label} "
                                                  f"to span {entity_span}")
                    else:
                        if not allow_many_entities_per_span_different_labels:
                            raise EntitySpanError(f"[{filename}]: span {entity_span} has already "
                                                  f"label {span2label[entity_span]},"
                                                  f"but tried to assign also label {entity_label}")
                else:
                    span2label[entity_span] = entity_label

                entity_matches = list(tokens_expression.finditer(expected_entity_pattern))

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
                        msg = "can not infer token id from span or span is a part of a token\n"
                        msg += f"absolute span: {token_span_abs}\n"
                        msg += f'entity token: <bos>{token}<eos>\n'
                        msg += f'corresponding text token: <bos>{text[s:e]}<eos>\n'
                        msg += f'context: {text[max(0, s - 50):s]}<bos>{text[s:e]}<eos>{text[e:e + 50]}'
                        raise EntitySpanError(msg)

                    # запись лейблов
                    if token.labels == no_labels:
                        token.labels = [label]
                    elif allow_nested_entities:
                        token_entity_labels = {l.split(ner_prefix_joiner)[-1] for l in token.labels}
                        if entity_label not in token_entity_labels:
                            token.labels.append(label)
                        else:
                            if allow_nested_entities_one_label:
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

    # создание инстанса класса Example
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


# TODO: проверить, работает ли
# TODO: создавать инстансы класса Event на уровне model.predict
def to_brat(
        examples: List[Example],
        output_dir: str,
        copy_texts: bool = False,
        collection_dir: str = None,
        write_mode: str = "a"
):
    assert write_mode in {"a", "w"}
    event_counter = defaultdict(int)
    filenames = set()
    for x in examples:
        filenames.add(x.filename)
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

    if copy_texts:
        assert collection_dir is not None
        for name in filenames:
            shutil.copy(os.path.join(collection_dir, f"{name}.txt"), output_dir)


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
