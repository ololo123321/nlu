from src.data.base import Example


def check_tokens_entities_alignment(example: Example):
    """
    * токены сущности соответствуют тексту сущности
    """
    for entity in example.entities:
        expected = _remove_spaces(entity.text)
        actual = ''
        for t in entity.tokens:
            actual += _remove_spaces(t.text)
        assert actual == expected, f"[{example.id}] [{entity.id}] {actual} != {expected}"


def check_flat_ner_markup(example: Example):
    """
    * каждый токен размечен
    * каждая сущность начинается с лейбла B-*
    * токены сущности соответствуют тексту сущности
    * пример начинается не в середине сущности
    """
    assert len(example.tokens) > 0, f"[{example.id}] no tokens"
    for t in example.tokens:
        assert t.label is not None, f"[{example.id}] token {t} has no label"
    assert example.tokens[0].label[0] != "I", f"[{example.id}] starts inside entity"
    for entity in example.entities:
        assert len(entity.tokens) > 0, f"[{example.id}] entity {entity.id} has no tokens"
        label = entity.tokens[0].label
        assert label[0] == "B", f"[{example.id}] entity {entity.id} starts with label {label}"
        expected = _remove_spaces(entity.text)
        actual = ''
        for t in entity.tokens:
            actual += _remove_spaces(t.text)
        assert actual == expected, f"[{example.id}] {actual} != {expected}"


def check_arcs(example: Example, one_child: bool = False, one_parent: bool = False):
    """
    * head и dep должны быть в множестве сущностей

    :param example:
    :param one_child: вершина может иметь не более одного исходящего ребра (coreference resolution)
    :param one_parent: вершина может иметь не более одного входящего ребра (dependency parsing)
    :return:
    """
    id2entity = {e.id: e for e in example.entities}

    # head и dep отношения содержатся с множетсве сущностей примера
    for arc in example.arcs:
        assert arc.head in id2entity.keys()
        assert arc.dep in id2entity.keys()

    if not (one_child or one_parent):
        return

    head2dep = {}
    dep2head = {}
    for arc in example.arcs:
        if one_child:
            if arc.head in head2dep.keys():
                head = id2entity[arc.head]
                dep_new = id2entity[arc.dep]
                dep_old = id2entity[head2dep[head.id]]
                msg = f'[{example.id}] head {head.id} <bos>{head.text}<eos> has already dep {dep_old.id} ' \
                    f'<bos>{dep_old.text}<eos>, but tried to assign dep {dep_new.id} <bos>{dep_new.text}<eos>'
                raise AssertionError(msg)
            else:
                head2dep[arc.head] = arc.dep
        if one_parent:
            if arc.dep in dep2head.keys():
                dep = id2entity[arc.dep]
                head_new = id2entity[arc.head]
                head_old = id2entity[dep2head[dep.id]]
                msg = f'[{example.id}] dep {dep.id} <bos>{dep.text}<eos> has already head {head_old.id} ' \
                    f'<bos>{head_old.text}<eos>, but tried to assign head {head_new.id} <bos>{head_new.text}<eos>'
                raise AssertionError(msg)
            else:
                dep2head[arc.dep] = arc.head


def check_split(chunk: Example, window: int, fixed_sent_pointers: bool = False):
    """
    * не должно быть пропусков предложений.
    * число предложений в куске может быть больше ширины окна только в том случае, если более ранние кандидаты на сплит
    проходили через сущность.
    """
    actual = {t.id_sent for t in chunk.tokens}
    id_sent_max = max(actual)
    id_sent_min = min(actual)

    expected = set(range(id_sent_min, id_sent_max + 1))
    assert actual == expected, f"[{chunk.id}] expected sent ids {expected}, but got {actual}"

    if fixed_sent_pointers:
        assert id_sent_max - id_sent_min < window, f"[{chunk.id}] expected example size <= {window} sentences, " \
            f"but got {id_sent_max - id_sent_min} sentences"

    if id_sent_max - id_sent_min >= window:
        id_sent_curr = chunk.tokens[0].id_sent
        sent_ids_to_check = set(range(id_sent_min + window, id_sent_max + 1))
        for t in chunk.tokens:
            if t.id_sent != id_sent_curr:
                id_sent_curr = t.id_sent
                if id_sent_curr in sent_ids_to_check:
                    assert t.label[0] == "I", f"[{chunk.id}] expected split " \
                        f"between sentences {id_sent_curr - 1} and {id_sent_curr}"


def _remove_spaces(s: str) -> str:
    s = s.replace(" ", "")
    s = s.replace("\xa0", "")
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    return s
