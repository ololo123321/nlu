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
            labels=no_labels.copy()
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