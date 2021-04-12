from src.data.base import Example


def check_tokens(example: Example):
    # число токенов сущности больше нуля
    entity_ids = set()
    for entity in example.entities:
        assert len(entity.tokens) > 0, f"[{example.id}] entity {entity.id} has no tokens!"
        entity_ids.add(entity.id)

    # пример не начинвается в середине сущности
    assert all(label[0] != "I" for label in example.tokens[0].labels), f"[{example.id}] contains only part of entity!"


def check_flat_ner_markup(example: Example):
    """
    число сущностей равно числу лейблов начала сущности
    """
    expected = len(example.entities)
    actual = 0
    for t in example.tokens:
        for label in t.labels:
            actual += label[0] == "B"
    assert actual == expected, \
        f"[{example.id}] number of entities ({expected}) does not match number of start tokens ({actual})"


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
                msg = f'[{example.id}] head {head.id} ({head.text}) has already dep {dep_old.id} ({dep_old.text}), ' \
                    f'but tried to assign dep {dep_new.id} ({dep_new.text})'
                raise AssertionError(msg)
            else:
                head2dep[arc.head] = arc.dep
        if one_parent:
            if arc.dep in dep2head.keys():
                dep = id2entity[arc.dep]
                head_new = id2entity[arc.head]
                head_old = id2entity[dep2head[dep.id]]
                msg = f'[{example.id}] dep {dep.id} ({dep.text}) has already head {head_old.id} ({head_old.text}), ' \
                    f'but tried to assign head {head_new.id} ({head_new.text})'
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
                    assert any(label[0] == "I" for label in t.labels), f"[{chunk.id}] expected split " \
                        f"between sentences {id_sent_curr - 1} and {id_sent_curr}"
