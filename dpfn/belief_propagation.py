"""Belief propagation for CRISP-like models."""
import numpy as np
from dpfn import constants, util
import numba
import time
from typing import Any, List, Optional, Tuple


@numba.njit
def adjust_matrices_map(
    A_matrix: np.ndarray,
    p1: float,
    forward_messages: np.ndarray,
    num_time_steps: int) -> np.ndarray:
  """Adjusts dynamics matrices based in messages from incoming contacts."""
  A_adjusted = np.copy(A_matrix)
  A_adjusted = np.ones((num_time_steps, 4, 4), dtype=np.double) * A_adjusted

  # First collate all incoming forward messages according to timestep
  log_probs = np.ones((num_time_steps)) * np.log(A_matrix[0][0])
  for row in forward_messages:
    _, user_me, timestep, p_inf_message = (
      int(row[0]), int(row[1]), int(row[2]), row[3])
    if user_me < 0:
      break
    # assert user_me == user, f"User {user_me} is not {user}"

    # Calculation
    add_term = np.log((p_inf_message * (1-p1) + (1-p_inf_message) * 1))
    log_probs[timestep] += add_term

  transition_prob = np.exp(log_probs)

  A_adjusted[:, 0, 0] = transition_prob
  A_adjusted[:, 0, 1] = 1. - transition_prob

  return A_adjusted


@numba.njit
def forward_backward_user(
    A_matrix: np.ndarray,
    p0: float,
    p1: float,
    user: int,
    backward_messages: np.ndarray,
    forward_messages: np.ndarray,
    num_time_steps: int,
    obs_messages: np.ndarray,
    start_belief: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Any], List[Any]]:
  """Does forward backward step for one user.

  Args:
    user: integer which is the user index (in absolute counting!!)
    map_backward_mesage: for each user, this is an array of size [CTC*T, 7]
      where the 7 columns are
      * user from
      * user to
      * timestep of contact
      * backward message S
      * backward message E
      * backward message I
      * backward message R
    map_backward_mesage: for each user, this is an array of size [CTC*T, 4]
      where the 4 columns are
      * user from
      * user to
      * timestep of contact
      * forward message as scalar
    start_belief: array/list of length 4, being start belief for SEIR states.

  Returns:
    marginal beliefs for this user after running bp, and the forward and
    backward messages that this user sends out.
  """
  # Default to start_belief as 1.-p0 in S and p0 in E
  if start_belief is None:
    start_belief = np.array([1.-p0, p0, 0., 0.])
  # assert start_belief.shape == (4,), f"Shape {start_belief.shape} is not [4] "

  # Collate backward messages
  mu_back_contact_log = np.zeros((num_time_steps, 4))  # Collate in matrix
  for row in backward_messages:
    if row[1] < 0:
      break
    # assert user == row[1], f"User {user} is not {row[1]}"
    _, timestep, message = int(row[0]), int(row[2]), row[3:]
    mu_back_contact_log[timestep] += np.log(message + 1E-12)
  mu_back_contact_log -= np.max(mu_back_contact_log)
  mu_back_contact = np.exp(mu_back_contact_log)
  mu_back_contact /= np.expand_dims(np.sum(mu_back_contact, axis=1), axis=1)

  # Clip messages in case of quantization
  mu_back_contact = np.clip(mu_back_contact, 0.0001, 0.9999)

  mu_f2v_forward = np.zeros((num_time_steps, 4))
  mu_f2v_backward = np.zeros((num_time_steps, 4))

  # Forward messages can be interpreted as modifying the dynamics matrix.
  # Therefore, we precompute these matrices using the incoming forward messages
  A_user = adjust_matrices_map(
    A_matrix, p1, forward_messages, num_time_steps)

  betas = np.zeros((num_time_steps, 4))
  # Move all messages forward
  mu_f2v_forward[0] = start_belief
  for t_now in range(1, num_time_steps):
    mu_f2v_forward[t_now] = A_user[t_now-1].T.dot(
      mu_f2v_forward[t_now-1] * obs_messages[t_now-1]
      * mu_back_contact[t_now-1])

  # Move all messages backward
  mu_f2v_backward[num_time_steps-1] = np.ones((4))
  for t_now in range(num_time_steps-2, -1, -1):
    mu_f2v_backward[t_now] = A_user[t_now].dot(
      mu_f2v_backward[t_now+1] * obs_messages[t_now+1]
      * mu_back_contact[t_now+1])

  # Collect marginal beliefs
  for t_now in range(num_time_steps):
    # TODO: do in logspace
    betas[t_now] = (mu_f2v_forward[t_now] * mu_f2v_backward[t_now]
                    * obs_messages[t_now] * mu_back_contact[t_now])

  # Calculate messages backward
  messages_send_back = -1 * np.ones((num_time_steps*constants.CTC, 7))
  num_bw = 0
  for row in forward_messages:
    user_backward, timestep_back = int(row[0]), int(row[2])
    p_message = float(row[3])
    if user_backward < 0:
      continue
    # assert timestep_back >= 0, (
    #   "Cannot send a message back on timestep 0. "
    #   "Probably some contact was defined for -1?")
    A_back = A_user[timestep_back]

    # This is the term that needs cancelling due to the forward message
    p_transition = A_back[0][0] / (p_message * (1-p1) + (1-p_message) * 1)

    # Cancel the terms in the two dynamics matrices
    A_back_0 = np.copy(A_back)
    A_back_1 = np.copy(A_back)
    A_back_0[0][0] = p_transition  # S --> S
    A_back_0[0][1] = 1. - p_transition  # S --> E
    A_back_1[0][0] = p_transition * (1-p1)  # S --> S
    A_back_1[0][1] = 1. - p_transition * (1-p1)  # S --> E

    # Calculate the SER terms and calculate the I term
    mess_SER = np.sum(
      A_back_0.dot(mu_f2v_backward[timestep_back+1]
                   * obs_messages[timestep_back+1])
      * mu_f2v_forward[timestep_back] * obs_messages[timestep_back])
    mess_I = np.sum(
      A_back_1.dot(mu_f2v_backward[timestep_back+1]
                   * obs_messages[timestep_back+1])
      * mu_f2v_forward[timestep_back] * obs_messages[timestep_back])
    message_back = np.array([mess_SER, mess_SER, mess_I, mess_SER]) + 1E-12
    # TODO (March 2022): figure out if this normalisation is correct
    message_back /= np.sum(message_back)
    # if np.any(np.logical_or(np.isinf(message_back), np.isnan(message_back))):
    #   logger.debug(f"Message back: \n {message_back}")
    #   logger.debug(f"mu_back_contact: \n {mu_back_contact}")

    array_back = np.array(
      [user, user_backward, timestep_back,
       message_back[0], message_back[1], message_back[2], message_back[3]],
      dtype=np.double)
    messages_send_back[num_bw] = array_back
    num_bw += 1

  # Calculate messages forward
  messages_send_forward = -1 * np.ones((num_time_steps*constants.CTC, 4))
  num_fw = 0
  for row in backward_messages:
    user_forward, timestep = int(row[0]), int(row[2])
    msg_back = row[3:]
    if user_forward < 0:
      continue
    # assert timestep < num_time_steps, (
    #   "Cannot send a message back on timestep <num_time_steps>. "
    #   "Probably some contact was defined for <num_time_steps>?")
    message_backslash = util.normalize(msg_back + 1E-12)
    # TODO: do in logspace
    message = (betas[timestep] / message_backslash)
    message /= np.sum(message)
    # if np.any(np.logical_or(np.isinf(message), np.isnan(message))):
    #   logger.debug(f"Message forward: \n {message}")
    #   logger.debug(f"Betas: \n {betas[timestep]}")
    #   logger.debug(f"mu_back_contact: \n {mu_back_contact[timestep]}")
    #   logger.debug(f"mu_f2v_forward: \n {mu_f2v_forward[timestep]}")
    #   logger.debug(f"mu_f2v_backward: \n {mu_f2v_backward[timestep]}")
    #   logger.debug(f"obs_messages: \n {obs_messages[timestep]}")
    #   logger.debug(f"Backward backslash: \n {message_backslash}")
    #   logger.debug(f"mu_f2v_forward: \n {mu_f2v_forward}")
    #   raise ValueError("NaN or INF in message")

    array_fwd = np.array([user, user_forward, timestep, message[2]])
    messages_send_forward[num_fw] = array_fwd
    num_fw += 1

  betas /= np.expand_dims(np.sum(betas, axis=1), axis=1)
  return betas, messages_send_back, messages_send_forward


def init_message_maps(
    contacts_all: List[constants.Contact],
    num_users: int,
    num_time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
  """Initialises the message maps."""
  # Put backward messages in hashmap, such that they can be overwritten when
  #   doing multiple iterations in loopy belief propagation
  max_num_contacts = num_time_steps * constants.CTC
  map_backward_message = -1 * np.ones((num_users, max_num_contacts, 7))
  map_forward_message = -1 * np.ones((num_users, max_num_contacts, 4))

  num_bw_message = np.zeros((num_users), dtype=np.int32)
  num_fw_message = np.zeros((num_users), dtype=np.int32)

  for contact in contacts_all:
    user_u = int(contact[0])
    user_v = int(contact[1])

    # Backward message:
    msg_bw = np.array(
      [user_v, user_u, contact[2], .25, .25, .25, .25], dtype=np.float32)
    map_backward_message[user_u][num_bw_message[user_u]] = msg_bw
    num_bw_message[user_u] += 1

    # Forward message:
    msg_fw = np.array([user_u, user_v, contact[2], 0.], dtype=np.float32)
    map_forward_message[user_v][num_fw_message[user_v]] = msg_fw
    num_fw_message[user_v] += 1
  return map_forward_message, map_backward_message


@numba.njit
def do_backward_forward_subset(
    user_interval: Tuple[int, int],
    A_matrix: np.ndarray,
    p0: float,
    p1: float,
    num_time_steps: int,
    obs_messages: np.ndarray,
    map_backward_message: np.ndarray,
    map_forward_message: np.ndarray,
    start_beliefs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Does forward backward on a subset of users in sequence.

  Note, the messages are appended, and thus not updated between users!
  """
  num_users_interval = user_interval[1] - user_interval[0]

  bp_beliefs_subset = np.zeros((num_users_interval, num_time_steps, 4))

  messages_backward_subset = np.zeros(
    (num_users_interval, num_time_steps*constants.CTC, 7))
  messages_forward_subset = np.zeros(
    (num_users_interval, num_time_steps*constants.CTC, 4))
  for user_id, user_run in enumerate(range(user_interval[0], user_interval[1])):
    start_belief = start_beliefs[user_id] if start_beliefs is not None else None
    bp_beliefs_subset[user_id], messages_back_user, messages_forward_user = (
      forward_backward_user(
        A_matrix, p0, p1, user_run,
        map_backward_message[user_id], map_forward_message[user_id],
        num_time_steps, obs_messages[user_id], start_belief))
    messages_backward_subset[user_id] = messages_back_user
    messages_forward_subset[user_id] = messages_forward_user

  return bp_beliefs_subset, messages_forward_subset, messages_backward_subset


@numba.njit
def do_backward_forward_and_message(
    A_matrix: np.ndarray,
    p0: float,
    p1: float,
    num_time_steps: int,
    obs_messages: np.ndarray,
    num_users: int,
    map_backward_message: np.ndarray,
    map_forward_message: np.ndarray,
    start_belief: Optional[np.ndarray] = None,
    quantization: Optional[int] = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
  """Runs forward and backward messages for one user and collates messages."""
  with numba.objmode(t0='f8'):
    t0 = time.time()

  bp_beliefs, msg_list_fwd, msg_list_bwd = do_backward_forward_subset(
    (0, num_users),
    A_matrix=A_matrix,
    p0=p0,
    p1=p1,
    num_time_steps=num_time_steps,
    obs_messages=obs_messages,
    map_backward_message=map_backward_message,
    map_forward_message=map_forward_message,
    start_beliefs=start_belief)

  with numba.objmode(t1='f8'):
    t1 = time.time()

  # TMP code
  max_num_contacts = num_time_steps * constants.CTC
  map_backward_message = -1 * np.ones((num_users, max_num_contacts, 7))
  map_forward_message = -1 * np.ones((num_users, max_num_contacts, 4))

  num_bw_message = np.zeros((num_users), dtype=np.int32)
  num_fw_message = np.zeros((num_users), dtype=np.int32)

  # Put backward messages generated by user_run in ndarray
  for i in range(num_users):
    # TODO: delete redundant for-loop
    for row in msg_list_bwd[i]:
      user_send, user_bwd, tstep = int(row[0]), int(row[1]), int(row[2])
      message = row[3:]

      if user_send < 0:
        break

      # Backward message:
      msg_bw = np.array(
        [user_send, user_bwd, tstep,
         message[0], message[1], message[2], message[3]], dtype=np.float32)
      map_backward_message[user_bwd][num_bw_message[user_bwd]] = msg_bw
      num_bw_message[user_bwd] += 1

  for i in range(num_users):
    # TODO: delete redundant for-loop
    for row in msg_list_fwd[i]:
      user_send, user_forward, tstep = int(row[0]), int(row[1]), int(row[2])
      message = float(row[3])

      if user_send < 0:
        break

      # Forward message:
      msg_fw = np.array(
        [user_send, user_forward, tstep, message], dtype=np.float32)
      map_forward_message[user_forward][num_fw_message[user_forward]] = msg_fw
      num_fw_message[user_forward] += 1

  map_backward_message[:, :, 3:] = util.quantize(
    map_backward_message[:, :, 3:], quantization)
  map_forward_message[:, :, 3] = util.quantize(
    map_forward_message[:, :, 3], quantization)

  # np.testing.assert_array_almost_equal(
  #   np.sum(bp_beliefs, axis=-1), 1., decimal=3)

  return bp_beliefs, map_backward_message, map_forward_message, t0, t1