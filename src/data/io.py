import os
import shutil
import re
import tqdm
from typing import List, Union, Pattern
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
    NestedNerError,
    NestedNerSingleEntityTypeError,
    RegexError
)


# TODO: адаптировать
# TODO: иметь возможность сохранть не только новые события, но и новые сущности
#  (то есть случай, при котором у инстансов класса Entity нет id)

def save_examples(
        examples: List[Example],
        output_dir: str,
        copy_texts: bool = False,
        collection_dir: str = None
):
    event_counter = defaultdict(int)
    filenames = set()
    for x in examples:
        filenames.add(x.filename)
        with open(os.path.join(output_dir, f"{x.filename}.ann"), "a") as f:
            events = {}
            # сущности
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
                args_str = args_str.strip()
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


def parse_collection(
        data_dir: str,
        n: int = None,
        fix_new_line_symbol: bool = True,
        ner_encoding: str = NerEncodings.BIO,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        tokens_pattern: Union[str, Pattern] = None,
        allow_nested_ner: bool = False,
        allow_nested_similar_entities: bool = False,
        ignore_bad_examples: bool = False
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
                fix_new_line_symbol=fix_new_line_symbol,
                ner_encoding=ner_encoding,
                ner_prefix_joiner=ner_prefix_joiner,
                tokens_expression=tokens_expression,
                allow_nested_ner=allow_nested_ner,
                allow_nested_similar_entities=allow_nested_similar_entities
            )
            examples.append(example)
        except (BadLineError, EntitySpanError, NestedNerError, NestedNerSingleEntityTypeError, RegexError) as e:
            err_name = type(e).__name__
            print(f"[{filename}] known error {err_name} occurred:")
            print(e)
            if ignore_bad_examples:
                print("example ignored due to flag ignore_bad_examples set to True")
                print("=" * 50)
                error_counts[err_name] += 1
            else:
                raise e
        except Exception as e:
            print(f"[{filename}] unknown error {type(e).__name__} occurred:")
            raise e
    print(f"successfully parsed {len(examples)} examples from {len(names_to_parse)} files.")
    print(f"error counts: {error_counts}")
    return examples


def parse_example(
        data_dir: str,
        filename: str,
        fix_new_line_symbol: bool = False,
        ner_encoding: str = NerEncodings.BIO,
        ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
        tokens_expression: Pattern = TOKENS_EXPRESSION,
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
                                              f"expected entity is {expected_entity_pattern}, "
                                              f"but got {actual_entity_pattern}")

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


def check_example(example: Example):
    """
    sanity check
    """
    # число токенов сущности больше нуля
    entity_ids = set()
    for entity in example.entities:
        assert len(entity.tokens) > 0, f"[{example.id}] entity {entity.id} has no tokens!"
        entity_ids.add(entity.id)

    # пример не начинвается в середине сущности
    assert example.tokens[0].labels[0][0] != "I", f"[{example.id}] contains only part of entity!"

    # число сущностей равно числу лейблов начала сущности
    expected = len(example.entities)
    actual = 0
    for t in example.tokens:
        l = t.labels[0]
        if l[0] == "B":
            actual += 1
    assert actual == expected, \
        f"[{example.id}] number os entities ({expected}) does not match number of start tokens ({actual})"

    # head и dep отношения содердатся с множетсве сущностей примера
    for arc in example.arcs:
        assert arc.head in entity_ids
        assert arc.dep in entity_ids
