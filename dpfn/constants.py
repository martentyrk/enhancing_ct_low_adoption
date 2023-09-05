"""Constants for the entire project."""

import numpy as np
from typing import Tuple, Union

# Definition of dp_method:
#
# Throughout the code, the variable 'dp_method' is used to determine which
# method is used to make the covidscore differentially private.
#
# The specified options are:
#
# DP=-1, no DP applied;
# DP=2, uses the global sensitivity;
# DP=3, uses the private geometric mean of the FN scores after equilibration
#         (doesn't work well);
# DP=4, uses the multiplicative sensitivity per row of FN
#         (but doesn't work well);
# DP=5, corresponds to using log-normal on user for a day;
# DP=6, corresponds to DP on message only, Gaussian Mechanism
#         in the logit domain;
# DP=7, corresponds to using the Gaussian mechanism in log-domain when doing FN
#         (note that this method introduces a bias in the covidscore);

CTC = 120  # Contact tracing capacity per day

colors = ['b', 'm', 'r', 'k']
state_names = ['S', 'E', 'I', 'R']

# Directory with ABM parameter files
ABM_HOME = "dpfn/data/abm_parameters/"

# States of ABM simulator:
# * UNINFECTED = 0
# * PRESYMPTOMATIC = 1
# * PRESYMPTOMATIC_MILD = 2
# * ASYMPTOMATIC = 3
# * SYMPTOMATIC = 4
# * SYMPTOMATIC_MILD = 5
# * HOSPITALISED = 6
# * CRITICAL = 7
# * RECOVERED = 8
# * DEATH = 9
# * QUARANTINED = 10
# * QUARANTINE_RELEASE = 11
# * TEST_TAKE = 12
# * TEST_RESULT = 13
# * CASE = 14
# * TRACE_TOKEN_RELEASE = 15
# * N_EVENT_TYPES = 16
state_to_seir = np.array(
  [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)

# Global type definitions

# (user_u, user_v, timestep, feature)
Contact = Union[Tuple[int, int, int, int], np.ndarray]

# (user_u, timestep, outcome)
Observation = Union[Tuple[int, int, int], np.ndarray]
