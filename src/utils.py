import os
from collections import defaultdict
import tensorflow as tf

from .preprocessing import Arc, Event, EventArgument


def compute_f1(preds, labels):
    """
    https://github.com/facebookresearch/SpanBERT/blob/10641ea3795771dd96e9e3e9ef0ead4f4f6a29d2/code/run_tacred.py#L245
    :param preds:
    :param labels:
    :return:
    """
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


def infer_entities_bounds(label_ids: tf.Tensor, bound_ids: tf.Tensor) -> tf.Tensor:
    """
    Вывод индексов первого или последнего токена сущностей
    :param label_ids: tf.Tensor of shape [N, T]
    :param bound_ids: tf.Tensor of shape [num_bound_ids] - айдишники, обозначающие начало или конец сущности
    :return: res: tf.Tensor of shape [num_entities_sum, 2], где num_entities_sum - общее число сущностей
             в батче. (i, j) - начало или конец сущности, где 0 <= i < N; 0 < j < T
    TODO: рассмотреть случай неизвестных лейблов токенов. вообще говоря, модель может в качестве первого
     или последнего лейбла сущности предсказать что-то другое (например, I_ORG вместо L_ORG)
    """
    labels_3d = tf.tile(label_ids[:, :, None], [1, 1, tf.shape(bound_ids)[0]])  # [N, T, num_bound_ids]
    mask_3d = tf.equal(labels_3d, bound_ids[None, None, :])  # [N, T, num_bound_ids]
    mask_2d = tf.reduce_any(mask_3d, axis=-1)  # [N, T]
    num_entities = tf.reduce_sum(tf.cast(mask_2d, tf.int32), axis=-1)  # [N]
    sequence_mask = tf.sequence_mask(num_entities)  # [N, num_entities_max]
    indices = tf.cast(tf.where(sequence_mask), tf.int32)  # [num_entities_sum, 2]
    updates = tf.cast(tf.where(mask_2d)[:, -1], tf.int32)  # [num_entities_sum]
    sequence_mask_shape = tf.shape(sequence_mask)
    res = tf.scatter_nd(indices, updates, shape=sequence_mask_shape)  # [N, num_entities_max]

    # Пусть число примеров = 3, число сущностей - 2
    num_examples = sequence_mask_shape[0]
    num_entities_max = sequence_mask_shape[1]
    x = tf.range(num_examples)  # [0, 1, 2]
    x = tf.tile(x[:, None], [1, num_entities_max])  # [[0, 0], [1, 1], [2, 2]]
    x = tf.reshape(x, [-1, 1])  # [[0], [0], [1], [1], [2], [2]]

    y = tf.reshape(res, [-1, 1])
    coords = tf.concat([x, y], axis=-1)
    return coords


def add_ones(x):
    ones = tf.ones_like(x[..., :1])
    x = tf.concat([x, ones], axis=-1)
    return x


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def save_baseline_predictions(examples, output_dir):
    """
    Бейзлайн:
    * если в предложении с событием есть компании, то привязать их все к событию
    * иначе, привязат к событию ближайшую компанию слева.

    ВАЖНО: примеры должны быть на уровне предложений!
    """
    event_counter = defaultdict(int)
    arc_counter = defaultdict(int)

    def get_event_id(filename):
        idx = event_counter[filename]
        event_counter[filename] += 1
        return idx

    def get_arc_id(filename):
        idx = arc_counter[filename]
        arc_counter[filename] += 1
        return idx

    id2example = {x.id: x for x in examples}
    ORG = "ORG"
    BANKRUPT = "Bankrupt"
    for x in examples:
        with open(os.path.join(output_dir, f"{x.filename}.ann"), "a") as f:
            events = {}
            org_ids = set()
            # исходные сущности
            for entity in x.entities:
                line = f"{entity.id}\t{entity.label} {entity.start_index} {entity.end_index}\t{entity.text}\n"
                f.write(line)
                if entity.is_event_trigger:
                    if entity.id not in events:
                        id_event = get_event_id(x.filename)
                        events[entity.id] = Event(
                            id=id_event,
                            id_trigger=entity.id,
                            label=entity.label,
                            arguments=None,
                        )
                elif entity.label == ORG:
                    org_ids.add(entity.id)

            # добавим рёбра
            if len(org_ids) > 0:
                for trigger in events.keys():
                    for id_entity in org_ids:
                        id_arc = get_arc_id(x.filename)
                        arc = Arc(
                            id=id_arc,
                            head=trigger,
                            dep=id_entity,
                            rel=BANKRUPT
                        )
                        x.arcs.append(arc)
            else:
                id_sent = int(x.id.split("_")[-1])
                if id_sent >= 1:
                    id_org_nearest = None
                    for j in range(id_sent - 1, -1, -1):
                        x_prev = id2example[f"{x.filename}_{j}"]
                        for entity in sorted(x_prev.entities, key=lambda e: e.end_index, reverse=True):
                            if entity.label == ORG:
                                id_org_nearest = entity.id
                                break
                        if id_org_nearest is not None:
                            for trigger in events.keys():
                                id_arc = get_arc_id(x.filename)
                                arc = Arc(
                                    id=id_arc,
                                    head=trigger,
                                    dep=id_org_nearest,
                                    rel=BANKRUPT
                                )
                                x.arcs.append(arc)
                            break

            # отношения
            for arc in x.arcs:
                if arc.head in events:
                    arg = EventArgument(id=arc.dep, role=arc.rel)
                    events[arc.head].arguments.add(arg)
                else:
                    line = f"R{arc.id}\t{arc.rel} Arg1:{arc.head} Arg2:{arc.dep}\n"
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
