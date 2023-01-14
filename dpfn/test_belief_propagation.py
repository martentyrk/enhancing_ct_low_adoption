"""Test functions for belief_propagation.py"""
from dpfn import belief_propagation
import numpy as np


def construct_test_problem():
  """Constructs data for testing problems."""
  alpha, beta = 0.001, 0.01

  p0, p1 = 0.01, 0.9
  g = h = 1/3
  A_matrix = np.array([
    [1-p0, p0, 0, 0],
    [0, 1-g, g, 0],
    [0, 0, 1-h, h],
    [0, 0, 0, 1]
  ])

  num_time_steps = 5
  observations_all = [
    (0, 2, 1),
    (1, 3, 1),
    ]
  contacts_all = [
    (0, 2, 1, 1),
    (1, 2, 1, 1),
  ]
  num_users = 3

  return (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all,
          num_users, num_time_steps)


def test_adjust_matrices_map():
  p0, p1 = 0.001, 0.1
  g = h = 1/5

  A_matrix = np.array([
    [1-p0, p0, 0, 0],
    [0, 1-g, g, 0],
    [0, 0, 1-h, h],
    [0, 0, 0, 1]
  ])

  # User 0 receives a forward message from user 1 at timestep 1
  forward_incoming_messages = np.array(
    [[1, 1, 1, .5], [-1, -1, -1, -1.]], dtype=np.float32)

  result = belief_propagation.adjust_matrices_map(
    A_matrix=A_matrix,
    p1=p1,
    forward_messages=forward_incoming_messages,
    num_time_steps=3)

  # The probably infected user 0 changes the p(s->s) for user 1
  p_ss = (1-p0)*(0.5*(1-p1) + 0.5)
  A_matrix_adj = np.array([
    [p_ss, 1.-p_ss, 0, 0],
    [0, 1-g, g, 0],
    [0, 0, 1-h, h],
    [0, 0, 0, 1]
  ])
  expected = np.stack((A_matrix, A_matrix_adj, A_matrix))

  # Message backward should go to user 1 at timestep 1
  np.testing.assert_array_almost_equal(result, expected)


def test_fward_bward_user():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  user_test = 0
  bp_beliefs_user, _, _ = (
    belief_propagation.forward_backward_user(
      A_matrix, p0, p1, user_test, map_backward_message[user_test],
      map_forward_message[user_test], num_time_steps, obs_messages[user_test]))

  # Compare to result with exact inference
  expected = np.array(
    [[0.745427799, 0.254572201, 0., 0.],
     [0.5550064083, 0.3186512833, 0.1263423084, 0.],
     [0.5494563443, 0.01180426, 0.4381102009, 0.0006291948],
     [0.5439617808, 0.0133640701, 0.2960082206, 0.1466659284],
     [0.538522163, 0.0143489979, 0.2017935038, 0.2453353353]]
  )
  np.testing.assert_array_almost_equal(bp_beliefs_user, expected, decimal=8)


def test_consistent_fward():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  user_test = 0
  _, list_bw, list_fw = (
    belief_propagation.forward_backward_user(
      A_matrix, p0, p1, user_test, map_backward_message[user_test],
      map_forward_message[user_test], num_time_steps, obs_messages[user_test]))

  assert np.all(list_bw < 0)
  np.testing.assert_almost_equal(list_fw[0][3], 0.12634230837716198, decimal=5)


def test_consistent_fward2():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all + [(0, 3, 1)]:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  user_test = 0
  _, list_bw, list_fw = (
    belief_propagation.forward_backward_user(
      A_matrix, p0, p1, user_test, map_backward_message[user_test],
      map_forward_message[user_test], num_time_steps, obs_messages[user_test]))

  assert np.all(list_bw < 0)
  np.testing.assert_almost_equal(list_fw[0][3], 0.27795103957754486, decimal=5)


def test_consistent_bward():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all + [(2, 3, 1)]:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))
  np.testing.assert_almost_equal(map_forward_message[2][0][0], 0)
  np.testing.assert_almost_equal(map_forward_message[2][1][0], 1)
  map_forward_message[2][0][-1] = .6
  map_forward_message[2][1][-1] = .7

  user_test = 2
  _, list_bw, list_fw = (
    belief_propagation.forward_backward_user(
      A_matrix, p0, p1, user_test, map_backward_message[user_test],
      map_forward_message[user_test], num_time_steps, obs_messages[user_test]))

  map_bw = {tuple(x[0:3].tolist()): x[3:] for x in list_bw}

  assert np.all(list_fw < 0)
  assert (2, 0, 1) in map_bw
  np.testing.assert_array_almost_equal(
    map_bw[(2, 0, 1)],
    np.array([0.22338414, 0.22338414, 0.32984757, 0.22338414]))
  assert (2, 1, 1) in map_bw
  np.testing.assert_array_almost_equal(
    map_bw[(2, 1, 1)],
    np.array([0.21366936, 0.21366936, 0.35899193, 0.21366936]))


def test_consistent_bward2():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all + [(2, 4, 1)]:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))
  np.testing.assert_almost_equal(map_forward_message[2][0][0], 0)
  np.testing.assert_almost_equal(map_forward_message[2][1][0], 1)
  map_forward_message[2][0][-1] = .6
  map_forward_message[2][1][-1] = .7

  user_test = 2
  _, list_bw, list_fw = (
    belief_propagation.forward_backward_user(
      A_matrix, p0, p1, user_test, map_backward_message[user_test],
      map_forward_message[user_test], num_time_steps, obs_messages[user_test]))

  map_bw = {tuple(x[0:3].tolist()): x[3:] for x in list_bw}

  assert np.all(list_fw < 0)
  assert (2, 0, 1) in map_bw
  np.testing.assert_array_almost_equal(
    map_bw[(2, 0, 1)],
    np.array([0.223115, 0.223115, 0.330656, 0.223115]))
  assert (2, 1, 1) in map_bw
  np.testing.assert_array_almost_equal(
    map_bw[(2, 1, 1)],
    np.array([0.213265, 0.213265, 0.360205, 0.213265]))


def test_fward_bward_messages_simul_receive2():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  for _ in range(5):
    (bp_beliefs, map_backward_message, map_forward_message, _, _) = (
      belief_propagation.do_backward_forward_and_message(
        A_matrix, p0, p1, num_time_steps, obs_messages,
        num_users, map_backward_message, map_forward_message, (0, num_users)))

  bp_beliefs /= np.sum(bp_beliefs, axis=-1, keepdims=True)

  # Results obtained from exact inference
  expected = np.array(
    [[[0.745427799, 0.254572201, 0., 0.],
      [0.5550064083, 0.3186512833, 0.1263423084, 0.],
      [0.5494563443, 0.01180426, 0.4381102009, 0.0006291948],
      [0.5439617808, 0.0133640701, 0.2960082206, 0.1466659284],
      [0.538522163, 0.0143489979, 0.2017935038, 0.2453353353]],

      [[0.7954540517, 0.2045459483, 0., 0.],
       [0.592953563, 0.3388644543, 0.0681819828, 0.],
       [0.4414820963, 0.3221599882, 0.2358523645, 0.000505551],
       [0.4370672753, 0.0107378826, 0.5505147275, 0.0016801146],
       [0.4326966025, 0.0115292611, 0.3705891125, 0.1851850238]],

      [[0.99, 0.01, 0., 0.],
       [0.9801, 0.0165666667, 0.0033333333, 0.],
       [0.8071972649, 0.1839471796, 0.0077444444, 0.0011111111],
       [0.7991252922, 0.1307034257, 0.0664786895, 0.0036925926],
       [0.7911340393, 0.0951268701, 0.0878869349, 0.0258521558]]])

  np.testing.assert_array_almost_equal(bp_beliefs, expected, decimal=8)


def test_fward_bward_messages_simul_send2():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  contacts_all = [
    (0, 2, 1, 1),
    (0, 1, 1, 1),
    ]

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  for _ in range(3):
    (bp_beliefs, map_backward_message, map_forward_message, _, _) = (
      belief_propagation.do_backward_forward_and_message(
        A_matrix, p0, p1, num_time_steps, obs_messages,
        num_users, map_backward_message, map_forward_message, (0, num_users)))
  bp_beliefs /= np.sum(bp_beliefs, axis=-1, keepdims=True)

  # Results obtained from exact inference
  expected = np.array([
    [[0.2807712408, 0.7192287592, 0., 0.],
     [0.209047527, 0.120022511, 0.670929962, 0.],
     [0.2069570517, 0.0044461673, 0.7852554963, 0.0033412847],
     [0.2048874812, 0.0050336821, 0.52498572, 0.2650931168],
     [0.2028386064, 0.0054046629, 0.351668374, 0.4400883567]],

    [[0.9229561605, 0.0770438395, 0., 0.],
     [0.8466827594, 0.1276359608, 0.0256812798, 0.],
     [0.1473794474, 0.7635944855, 0.0888356473, 0.0001904198],
     [0.1459056529, 0.0164609286, 0.8370005901, 0.0006328284],
     [0.1444465964, 0.012433009, 0.5634873696, 0.2796330251]],

    [[0.99, 0.01, 0., 0.],
     [0.9801, 0.0165666667, 0.0033333333, 0.],
     [0.3843965959, 0.6067478485, 0.0077444444, 0.0011111111],
     [0.3805526299, 0.4083425317, 0.2074122458, 0.0036925926],
     [0.3767471036, 0.2760338807, 0.2743890078, 0.0728300079]]])

  np.testing.assert_array_almost_equal(bp_beliefs, expected, decimal=8)


def test_fward_bward_messages_simul_send3():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  contacts_all = [
    (0, 1, 1, 1),
    (0, 2, 1, 1),
    (0, 3, 1, 1),
    ]
  num_users += 1

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  # Note: will not work in two rounds due to concurrency among slice
  for _ in range(3):
    (bp_beliefs, map_backward_message, map_forward_message, _, _) = (
      belief_propagation.do_backward_forward_and_message(
        A_matrix, p0, p1, num_time_steps, obs_messages,
        num_users, map_backward_message, map_forward_message, (0, num_users)))

  bp_beliefs /= np.sum(bp_beliefs, axis=-1, keepdims=True)

  # Results obtained from exact inference
  expected = np.array([
    [[0.2807712408, 0.7192287592, 0., 0.],
     [0.209047527, 0.120022511, 0.670929962, 0.],
     [0.2069570517, 0.0044461673, 0.7852554963, 0.0033412847],
     [0.2048874812, 0.0050336821, 0.52498572, 0.2650931168],
     [0.2028386064, 0.0054046629, 0.351668374, 0.4400883567]],

    [[0.9229561605, 0.0770438395, 0., 0.],
     [0.8466827594, 0.1276359608, 0.0256812798, 0.],
     [0.1473794474, 0.7635944855, 0.0888356473, 0.0001904198],
     [0.1459056529, 0.0164609286, 0.8370005901, 0.0006328284],
     [0.1444465964, 0.012433009, 0.5634873696, 0.2796330251]],

    [[0.99, 0.01, 0., 0.],
     [0.9801, 0.0165666667, 0.0033333333, 0.],
     [0.3843965959, 0.6067478485, 0.0077444444, 0.0011111111],
     [0.3805526299, 0.4083425317, 0.2074122458, 0.0036925926],
     [0.3767471036, 0.2760338807, 0.2743890078, 0.0728300079]],

    [[0.99, 0.01, 0., 0.],
     [0.9801, 0.0165666667, 0.0033333333, 0.],
     [0.3843965959, 0.6067478485, 0.0077444444, 0.0011111111],
     [0.3805526299, 0.4083425317, 0.2074122458, 0.0036925926],
     [0.3767471036, 0.2760338807, 0.2743890078, 0.0728300079]]])

  np.testing.assert_array_almost_equal(bp_beliefs, expected, decimal=8)


def test_fward_bward_messages_simul_receive3():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  observations_all = [
    (0, 2, 1),
    (1, 3, 1),
    (2, 3, 1),
    ]

  contacts_all = [
    (0, 3, 1, 1),
    (1, 3, 1, 1),
    (2, 3, 1, 1),
    ]
  num_users += 1

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  bp_beliefs = np.zeros((num_users, num_time_steps, 4))
  for _ in range(2):
    (bp_beliefs, map_backward_message, map_forward_message, _, _) = (
      belief_propagation.do_backward_forward_and_message(
        A_matrix, p0, p1, num_time_steps, obs_messages,
        num_users, map_backward_message, map_forward_message, (0, num_users)))

  bp_beliefs /= np.sum(bp_beliefs, axis=-1, keepdims=True)

  # Results obtained from exact inference
  expected = np.array([
    [[0.745427799, 0.254572201, 0., 0.],
     [0.5550064083, 0.3186512833, 0.1263423084, 0.],
     [0.5494563443, 0.01180426, 0.4381102009, 0.0006291948],
     [0.5439617808, 0.0133640701, 0.2960082206, 0.1466659284],
     [0.538522163, 0.0143489979, 0.2017935038, 0.2453353353]],

    [[0.7954540517, 0.2045459483, 0., 0.],
     [0.592953563, 0.3388644543, 0.0681819828, 0.],
     [0.4414820963, 0.3221599882, 0.2358523645, 0.000505551],
     [0.4370672753, 0.0107378826, 0.5505147275, 0.0016801146],
     [0.4326966025, 0.0115292611, 0.3705891125, 0.1851850238]],

    [[0.7954540517, 0.2045459483, 0., 0.],
     [0.592953563, 0.3388644543, 0.0681819828, 0.],
     [0.4414820963, 0.3221599882, 0.2358523645, 0.000505551],
     [0.4370672753, 0.0107378826, 0.5505147275, 0.0016801146],
     [0.4326966025, 0.0115292611, 0.3705891125, 0.1851850238]],

    [[0.99, 0.01, 0., 0.],
     [0.9801, 0.0165666667, 0.0033333333, 0.],
     [0.7576645859, 0.2334798586, 0.0077444444, 0.0011111111],
     [0.75008794, 0.1632298849, 0.0829895825, 0.0036925926],
     [0.7425870606, 0.1163208027, 0.10973635, 0.0313557868]]])

  np.testing.assert_array_almost_equal(bp_beliefs, expected, decimal=8)


def test_fward_bward_messages_simul_send3_quantized():
  (p0, p1, alpha, beta, A_matrix, observations_all, contacts_all, num_users,
   num_time_steps) = construct_test_problem()

  contacts_all = [
    (0, 1, 1, 1),
    (0, 2, 1, 1),
    (0, 3, 1, 1),
    ]
  num_users += 1

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4))
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users), num_time_steps))

  # Note: will not work in two rounds due to concurrency among slice
  for _ in range(3):
    (bp_beliefs, map_backward_message, map_forward_message, _, _) = (
      belief_propagation.do_backward_forward_and_message(
        A_matrix, p0, p1, num_time_steps, obs_messages,
        num_users, map_backward_message, map_forward_message,
        user_interval=(0, num_users), quantization=256))

  bp_beliefs /= np.sum(bp_beliefs, axis=-1, keepdims=True)

  # Results obtained from exact inference
  expected = np.array([
    [[0.2807712408, 0.7192287592, 0., 0.],
     [0.209047527, 0.120022511, 0.670929962, 0.],
     [0.2069570517, 0.0044461673, 0.7852554963, 0.0033412847],
     [0.2048874812, 0.0050336821, 0.52498572, 0.2650931168],
     [0.2028386064, 0.0054046629, 0.351668374, 0.4400883567]],

    [[0.9229561605, 0.0770438395, 0., 0.],
     [0.8466827594, 0.1276359608, 0.0256812798, 0.],
     [0.1473794474, 0.7635944855, 0.0888356473, 0.0001904198],
     [0.1459056529, 0.0164609286, 0.8370005901, 0.0006328284],
     [0.1444465964, 0.012433009, 0.5634873696, 0.2796330251]],

    [[0.99, 0.01, 0., 0.],
     [0.9801, 0.0165666667, 0.0033333333, 0.],
     [0.3843965959, 0.6067478485, 0.0077444444, 0.0011111111],
     [0.3805526299, 0.4083425317, 0.2074122458, 0.0036925926],
     [0.3767471036, 0.2760338807, 0.2743890078, 0.0728300079]],

    [[0.99, 0.01, 0., 0.],
     [0.9801, 0.0165666667, 0.0033333333, 0.],
     [0.3843965959, 0.6067478485, 0.0077444444, 0.0011111111],
     [0.3805526299, 0.4083425317, 0.2074122458, 0.0036925926],
     [0.3767471036, 0.2760338807, 0.2743890078, 0.0728300079]]])

  np.testing.assert_array_almost_equal(bp_beliefs, expected, decimal=2)
