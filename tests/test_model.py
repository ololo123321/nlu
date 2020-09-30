import pytest
import numpy as np
import tensorflow as tf
from src.utils import infer_entities_bounds


sess = tf.InteractiveSession()
BOUND_IDS = tf.constant([1, 2, 5])


@pytest.mark.parametrize("label_ids, expected", [
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
        ])
    )
])
def test_infer_entities_bounds(label_ids, expected):
    actual = infer_entities_bounds(label_ids=label_ids, bound_ids=BOUND_IDS)
    actual = actual.eval()
    assert np.allclose(actual, expected)
