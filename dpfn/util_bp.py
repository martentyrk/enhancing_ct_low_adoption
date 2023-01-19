"""Utility functions for belief propagation inference."""

from dpfn import constants
import numba
import numpy as np


@numba.njit
def flip_message_send(
    message_list: np.ndarray,
    num_users: int,
    num_time_steps: int,
    do_bwd: bool = False) -> np.ndarray:
  """Sorts the messages by receiving user.

  The array is filled with '-1' for non-existing messages, up to a shape of
  num_time_steps*constants.CTC

  Args:
    message_list: List of messages generated by user_send
    user_interval: Interval of users to consider
    num_time_steps: Number of time steps
    do_bwd: indicator for backward messages
      if true, send backward messages of 7 elements
        [user_send, user_receive, tstep, 4*message]
      if false, send forward messages of 4 elements
        [user_send, user_receive, tstep, message]
  """
  num_elements = 7 if do_bwd else 4
  max_num_contacts = num_time_steps * constants.CTC
  map_messages = -1 * np.ones(
    (num_users, max_num_contacts, num_elements), dtype=np.float32)

  # Maintains how many non-null elements there are for a user
  num_messages = np.zeros((num_users), dtype=np.int32)

  # Put messages generated by user_run in ndarray
  for message_per_user in message_list:
    for row in message_per_user:
      # Unpack message to be sent
      user_send, user_receive, tstep = int(row[0]), int(row[1]), int(row[2])
      message = row[3:]  # Fwd message is 1 float, bwd message is 4 floats

      # Assumes arrays are filled with -1 for non-existing messages
      if user_send < 0:
        break

      # Construct message:
      msg_pre_array = np.array(
        [user_send, user_receive, tstep], dtype=np.float32)
      msg_post_array = message.astype(np.float32)
      msg_array = np.concatenate((msg_pre_array, msg_post_array), axis=0)

      # User receive id, relative to interval
      map_messages[user_receive][num_messages[user_receive]] = msg_array
      num_messages[user_receive] += 1
  return map_messages


@numba.njit(parallel=True)
def collapse_null_messages(
    messages: np.ndarray, num_proc: int, num_users_int: int, num_contacts: int,
    num_elements: int) -> np.ndarray:
  """Collapses the messages generated by flip_message_send.

  Removes the null messages and returns a 3D array of shape
    [num_users, num_time_steps*CTC, num_elements]
  """
  out = -1 * np.ones(
    (num_users_int, num_contacts, num_elements), dtype=np.single)
  num_put = np.zeros((num_users_int), dtype=np.int32)

  for user in numba.prange(num_users_int):  # pylint: disable=not-an-iterable
    for i in range(num_proc):
      for j in range(num_contacts):
        if messages[i][user][j][0] >= 0:
          out[user][num_put[user]] = messages[i][user][j]
          num_put[user] += 1
  return out
