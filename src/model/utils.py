from typing import Tuple, Dict
import tensorflow as tf


def get_labels_mask(labels_2d: tf.Tensor, values: tf.Tensor, sequence_len: tf.Tensor) -> tf.Tensor:
    """
    Получение маски: mask[i, j] = any(labels[i, j] == v for v in values)
    :param labels_2d: tf.Tensor of shape [N, T] and type tf.int32 - label ids
    :param values: tf.Tensor of shape [num_ids] and type tf.int32 - start label ids. может быть пустым
    :param sequence_len: tf.Tensor of shape [N] and type tf.int32 - sequence lengths
    :return: tf.Tensor of shape [N, T] and type tf.bool
    """
    labels_3d = tf.tile(labels_2d[:, :, None], [1, 1, tf.shape(values)[0]])  # [N, T, num_bound_ids]
    mask_3d = tf.equal(labels_3d, values[None, None, :])  # [N, T, num_bound_ids]
    mask_2d = tf.reduce_any(mask_3d, axis=-1)  # [N, T]
    mask_2d = tf.logical_and(mask_2d, tf.sequence_mask(sequence_len))  # [N, T]
    return mask_2d


def get_padded_coords(mask_2d: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Получение кординат элементов True с паддингом (0, 0).
    Для mask_2d
    [[False, True,  False],
     [True,  False, True],
     [False, False, False]]
    будут возвращены следующие координаты:
    [
     [[0, 1], [0, 0]],
     [[1, 0], [1, 2]],
     [[0, 0], [0, 0]]
    ]
    В случае отсутствия элементов True возвращается нулевой тензор размерности [mask_2d.shape[0], 2]

    :param mask_2d: tf.Tensor of shape [batch_size, maxlen] and type tf.bool
    :return: coords: tf.Tensor of shape [batch_size, num_elements_max, 2],
             где num_elements_max - наибольее число элементов True в строке mask_2d.
             (i, j) - начало или конец сущности, где 0 <= i < N; 0 <= j < T
    """
    # вывод координаты y
    num_elements = tf.reduce_sum(tf.cast(mask_2d, tf.int32), axis=-1)  # [N]
    sequence_mask = tf.sequence_mask(num_elements)  # [N, num_elements_max]
    indices = tf.cast(tf.where(sequence_mask), tf.int32)  # [num_elements_sum, 2]
    updates = tf.cast(tf.where(mask_2d)[:, -1], tf.int32)  # [num_elements_sum]
    sequence_mask_shape = tf.shape(sequence_mask)
    y_coord = tf.scatter_nd(indices, updates, shape=sequence_mask_shape)  # [N, num_elements_max]
    # res.dtype = updates.dtype

    # вывод координаты x
    # Пусть число примеров = 3, максимальное число элементов = 2
    batch_size = sequence_mask_shape[0]
    num_elements_max = sequence_mask_shape[1]
    x_coord = tf.range(batch_size, dtype=tf.int32)  # [N], [0, 1, 2]
    x_coord = tf.tile(x_coord[:, None], [1, num_elements_max])  # [N, num_elements_max], [[0, 0], [1, 1], [2, 2]]

    # объединение координат x и y
    coords = tf.concat([x_coord[:, :, None], y_coord[:, :, None]], axis=-1)  # [N, num_elements_max, 2]

    # фейковые координаты в случае отсутствия сущностей
    coords_dummy = tf.zeros([batch_size, 1, 2], dtype=tf.int32)
    cond = tf.equal(tf.reduce_max(num_elements), 0)
    coords = tf.cond(cond, true_fn=lambda: coords_dummy, false_fn=lambda: coords)

    return coords, num_elements


def get_batched_coords_from_labels(
        labels_2d: tf.Tensor,
        values: tf.Tensor,
        sequence_len: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    mask_2d = get_labels_mask(labels_2d=labels_2d, values=values, sequence_len=sequence_len)
    return get_padded_coords(mask_2d=mask_2d)


def get_entity_embeddings(
        x: tf.Tensor,
        d_model: int,
        emb_type: int,
        entity_start_ids: tf.Tensor,
        entity_end_ids: tf.Tensor = None
) -> tf.Tensor:
    """
    Векторизация сущностей. Предполагается, что границы сущностей известны.

    :param x:
    :param d_model:
    :param emb_type:
    :param entity_start_ids:
    :param entity_end_ids:
    :return:
    """
    batch_size = tf.shape(x)[0]

    if emb_type == 0:
        x_span = tf.gather_nd(x, entity_start_ids)  # [N * num_entities, D]
    elif emb_type == 1:
        assert entity_end_ids is not None
        one = tf.tile([[0, 1]], [tf.shape(entity_start_ids)[0], 1])
        x_i = tf.gather_nd(x, entity_start_ids)  # [N * num_entities, D]
        x_i_minus_one = tf.gather_nd(x, entity_start_ids - one)  # [N * num_entities, D]
        x_j = tf.gather_nd(x, entity_end_ids)  # [N * num_entities, D]
        x_j_plus_one = tf.gather_nd(x, entity_end_ids + one)  # [N * num_entities, D]

        d_model_half = d_model // 2
        x_start = x_j - x_i_minus_one
        x_start = x_start[..., :d_model_half]
        x_end = x_i - x_j_plus_one
        x_end = x_end[..., d_model_half:]

        x_span = tf.concat([x_start, x_end], axis=-1)  # [N * num_entities, D]
    else:
        raise ValueError(f"expected span_emb type in {{0, 1}}, got {emb_type}")

    x_span = tf.reshape(x_span, [batch_size, -1, d_model])  # [N, num_entities, D]

    return x_span


def add_ones(x: tf.Tensor) -> tf.Tensor:
    ones = tf.ones_like(x[..., :1])
    x = tf.concat([x, ones], axis=-1)
    return x


def noam_scheme(init_lr: int, global_step: int, warmup_steps: int = 4000):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
