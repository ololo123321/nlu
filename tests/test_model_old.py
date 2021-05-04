# import pytest
# import numpy as np
# import tensorflow as tf
#
# from src.model.utils import infer_entities_bounds
#
#
# # TODO: актуализировать
#
# sess = tf.Session()
# BOUND_IDS = tf.constant([1, 2, 5])
#
#
# @pytest.mark.parametrize("label_ids, expected_coords, expected_num_entities", [
#     pytest.param(
#         tf.constant([
#             [0, 0, 0],
#             [0, 0, 0]
#         ]),
#         np.array([
#             [0, 0],
#             [0, 0]
#         ]),
#         np.array([0, 0]),
#         id="no entities"
#     ),
#     pytest.param(
#         tf.constant([
#             [0, 0],
#             [0, 1]
#         ]),
#         np.array([
#             [0, 0],
#             [1, 1]
#         ]),
#         np.array([0, 1]),
#         id="single entity"
#     ),
#     pytest.param(
#         tf.constant([
#             [0, 1, 3, 5, 0],
#             [0, 0, 2, 4, 0],
#             [0, 0, 5, 6, 0],
#             [0, 0, 0, 0, 0]
#         ]),
#         np.array([
#             [0, 1],
#             [0, 3],
#             [1, 2],
#             [1, 0],
#             [2, 2],
#             [2, 0],
#             [3, 0],
#             [3, 0]
#         ]),
#         np.array([2, 1, 1, 0]),
#         id="many entities"
#     )
# ])
# def test_infer_entities_bounds(label_ids, expected_coords, expected_num_entities):
#     coords, num_entities = infer_entities_bounds(label_ids=label_ids, bound_ids=BOUND_IDS)
#
#     coords = sess.run(coords)
#     # print(coords)
#     assert np.allclose(coords, expected_coords)
#
#     num_entities = sess.run(num_entities)
#     # print(num_entities)
#     assert np.allclose(num_entities, expected_num_entities)
