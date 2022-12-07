"""Unit tests for simulator.py"""

from dpfn import simulator


def test_window_cut():

  contacts = [
    (0, 23, 0, [1]),
    (0, 23, 1, [1]),
    (0, 23, 2, [1]),
    (0, 23, 3, [1]),
    (0, 23, 4, [1]),
    (0, 23, 5, [1]),
    (0, 23, 6, [1]),
    (0, 23, 7, [1]),
    (0, 23, 8, [1]),
    (0, 23, 9, [1]),
  ]

  sim = simulator.CRISPSimulator(num_time_steps=30, num_users=100, params={})
  sim.init_day0(contacts=contacts)

  assert sim.get_contacts() == contacts
  assert len(sim.get_contacts()) == 10

  sim.set_window(days_offset=1)

  assert sim.get_contacts() == contacts[:-1]
  assert len(sim.get_contacts()) == 9

  sim.set_window(days_offset=5)

  assert sim.get_contacts() == contacts[:-5]
  assert len(sim.get_contacts()) == 5

  sim.set_window(days_offset=6)

  assert sim.get_contacts() == contacts[:-6]
  assert len(sim.get_contacts()) == 4
