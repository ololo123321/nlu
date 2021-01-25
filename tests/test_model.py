import pytest
import numpy as np
import tensorflow as tf
from src.utils import infer_entities_bounds


sess = tf.Session()
BOUND_IDS = tf.constant([1, 2, 5])


@pytest.mark.parametrize("label_ids, expected_coords, expected_num_entities", [
    pytest.param(
        tf.constant([
            [0, 1, 3, 5, 0],
            [0, 0, 2, 4, 0],
            [0, 0, 5, 6, 0],
            [0, 0, 0, 0, 0]
        ]),
        np.array([
            [0, 1],
            [0, 3],
            [1, 2],
            [1, 0],
            [2, 2],
            [2, 0],
            [3, 0],
            [3, 0]
        ]),
        np.array([2, 1, 1, 0])
    )
])
def test_infer_entities_bounds(label_ids, expected_coords, expected_num_entities):
    coords, num_entities = infer_entities_bounds(label_ids=label_ids, bound_ids=BOUND_IDS)
    actual_coords = sess.run(coords)
    assert np.allclose(actual_coords, expected_coords)
    actual_num_entities = sess.run(num_entities)
    assert np.allclose(actual_num_entities, expected_num_entities)
