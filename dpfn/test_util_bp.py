"""Test functions for util_bp.py."""

from dpfn import util_bp
import numpy as np


def test_flip_message_send():
  msg_list = np.array([[
    [0, 1, 0, 0.1],
    [0, 2, 0, 0.2],
    [0, 3, 0, 0.3],
    [0, 4, 0, 0.4],
    [0, 5, 0, 0.5],
    [-1, -1, -1, -1]], [
    [1, 0, 0, 0.7],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1]]], dtype=np.float32)
  msg_list = np.concatenate(
    (msg_list, -1 * np.ones((4, 6, 4), dtype=np.float32)))
  result = util_bp.flip_message_send(
    msg_list, 6, 5, do_bwd=False)

  np.testing.assert_array_almost_equal(result.shape, (6, 500, 4))
  np.testing.assert_array_almost_equal(result[0, 0, :], [1, 0, 0, 0.7])
  np.testing.assert_array_almost_equal(result[1, 0, :], [0, 1, 0, 0.1])
  np.testing.assert_array_almost_equal(result[2, 0, :], [0, 2, 0, 0.2])


def test_collapse_null_messages():
  num_proc = 2
  num_users = 5
  max_num_messages = 20
  msg_0_0 = np.array(
    [[5, 0, 0, 0.1],
     [7, 0, 0, 0.2],
     [9, 0, 0, 0.3],
     [11, 0, 0, 0.4],
     [13, 0, 0, 0.5],
     [-1, -1, -1, -1],
     [-1, -1, -1, -1],
     [-1, -1, -1, -1],], dtype=np.float32)
  msg_0_1 = np.array(
    [[29, 1, 0, 0.1],
     [-1, -1, -1, -1],], dtype=np.float32)
  msg_1_1 = np.array(
    [[15, 1, 0, 0.1],
     [19, 1, 0, 0.3],
     [17, 1, 0, 0.2],
     [1, 1, 0, 0.4],
     [-1, -1, -1, -1],
     [-1, -1, -1, -1],
     [-1, -1, -1, -1],
     [-1, -1, -1, -1],], dtype=np.float32)

  msg_all = -1 * np.ones((num_proc, num_users, max_num_messages, 4))
  msg_all[0, 0, :8, :] = msg_0_0
  msg_all[0, 1, :2, :] = msg_0_1
  msg_all[1, 1, :8, :] = msg_1_1

  result = util_bp.collapse_null_messages(
    msg_all, num_proc, num_users, max_num_messages, 4)

  np.testing.assert_array_almost_equal(
    result.shape, (num_users, max_num_messages, 4))
  np.testing.assert_array_almost_equal(
    result[0, :8, :], msg_0_0)
  np.testing.assert_array_almost_equal(
    result[1, 0, :], [29, 1, 0, 0.1])
  np.testing.assert_array_almost_equal(
    result[1, 1:9, :], msg_1_1)
