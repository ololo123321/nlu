import tensorflow as tf


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
    :param x: tf.Tensor of shape [N, T, D]
    :param label_ids: tf.Tensor of shape [N, T]
    :param bound_ids: tf.Tensor of shape [num_bound_ids] - айдишники, обозначающие начало или конец сущности
    :return: coords: tf.Tensor of shape [num_entities_sum, 2], где num_entities_sum - общее число сущностей
             в батче. (i, j) - начало или конец сущности, где 0 <= i < N; 0 < j < T
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
    x_coord = tf.range(num_examples)  # [0, 1, 2]
    x_coord = tf.tile(x_coord[:, None], [1, num_entities_max])  # [[0, 0], [1, 1], [2, 2]]
    x_coord = tf.reshape(x_coord, [-1, 1])  # [[0], [0], [1], [1], [2], [2]]

    y_coord = tf.reshape(res, [-1, 1])  # [N * num_entities_max, 1]
    coords = tf.concat([x_coord, y_coord], axis=-1)  # [N * num_entities_max, 2]

    return coords


def add_ones(x):
    ones = tf.ones_like(x[..., :1])
    x = tf.concat([x, ones], axis=-1)
    return x


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def check_entities_spans(examples, span_emb_type):
    """
    Дополнительная проверка примеров в скрипте обучения и инференса
    :param examples:
    :param span_emb_type: 0 - вектор первого токена, 1 - выводится из крайних и сосендних векторов сущности
    (см. RelationExtractor._get_entity_embeddings)
    :return:
    """
    for x in examples:
        for entity in x.entities:
            actual = ' '.join(x.tokens[entity.start_token_id:entity.end_token_id + 1])
            expected = ' '.join(entity.tokens)
            assert actual == expected
            if span_emb_type == 0:
                assert entity.start_token_id >= 0
            elif span_emb_type == 1:
                assert entity.start_token_id >= 1
            else:
                raise
