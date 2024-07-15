"""Inference methods for contact-graphs."""
from dpfn import constants, logger, util
import numba
import numpy as np
import os  # pylint: disable=unused-import
import time
from typing import Any, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


class Dset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (ndarray or Tensor): Array or Tensor containing input data.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@numba.njit(['float32[:](float32[:])', 'float64[:](float64[:])'])
def softmax(x):
    y = x - np.max(x)
    return np.exp(y)/np.sum(np.exp(y))


@numba.njit(parallel=True)
def fn_step_wrapped(
    user_interval: Tuple[int, int],
    seq_array_hot: np.ndarray,
    log_c_z_u: np.ndarray,
    log_A_start: np.ndarray,
    p_infected_matrix: np.ndarray,
    num_time_steps: int,
    probab0: float,
    probab1: float,
    past_contacts_array: np.ndarray,
    app_users: np.ndarray,
    prev_z_states: np.ndarray,
    clip_lower: float = -1.,
    clip_upper: float = 10000.,
    quantization: int = -1,
    infection_prior: float = -1,
    non_app_user_ids: np.ndarray = np.array([]),
):
    """Wraps one step of Factorised Neighbors over a subset of users.

    Args:
      user_interval: tuple of (user_start, user_end)
      seq_array_hot: array in [num_time_steps, 4, num_sequences]
      log_c_z_u: array in [num_users_int, num_sequences], C-terms according to
        CRISP paper
      log_A_start: array in [num_sequences], A-terms according to CRISP paper
      p_infected_matrix: array in [num_users, num_time_steps]
      num_time_steps: number of time steps
      probab0: probability of transitioning S->E
      probab1: probability of transmission given contact
      clip_lower: lower margin for clipping in preparation for DP calculations
      clip_upper: upper margin for clipping in preparation for DP calculations
      past_contacts_array: iterator with elements (timestep, user_u, features)
      dp_method: DP method to use, explanation in constants.py, value of -1 means
        no differential privacy applied
      epsilon_dp: epsilon for DP
      delta_dp: delta for DP
      a_rdp: alpha parameter for Renyi Differential Privacy
      quantization: number of quantization levels
    """
    # Only timer function in object-mode
    with numba.objmode(t0='f8'):
        t0 = time.time()

    # Apply quantization
    if quantization > 0:
        p_infected_matrix = util.quantize_floor(
            p_infected_matrix, quantization)

    p_infected_matrix = p_infected_matrix.astype(np.float32)
    # Apply upper clipping
    if clip_upper < 1.0:
        p_infected_matrix = np.minimum(
            p_infected_matrix, np.float32(clip_upper))

    # Apply lower clipping
    if clip_lower > 0.0:
        p_infected_matrix = np.maximum(
            p_infected_matrix, np.float32(clip_lower))

    interval_num_users = user_interval[1] - user_interval[0]
    post_exps = np.zeros(
        (interval_num_users, num_time_steps, 4), dtype=np.float32)
    num_days_s = np.sum(seq_array_hot[:, 0], axis=0).astype(np.int32)

    assert np.all(np.sum(seq_array_hot, axis=1) == 1), (
        "seq_array_hot is expected as one-hot array")

    seq_array_hot = seq_array_hot.astype(np.single)
    num_sequences = seq_array_hot.shape[2]

    # inf_mean = np.mean(prev_z_states)
    # inf_std = np.std(prev_z_states)
    # util.gaussian_imputation(
    #   inf_mean,
    #   inf_std,
    #   p_infected_matrix,
    #   non_app_user_ids
    # )

    if infection_prior != -1.:
        p_infected_matrix[non_app_user_ids, :] = infection_prior

    for i in numba.prange(interval_num_users):  # pylint: disable=not-an-iterable
        if app_users[i]:
            d_term, d_no_term = util.precompute_d_penalty_terms_fn2(
                q_marginal_infected=p_infected_matrix,
                p0=probab0,
                p1=probab1,
                past_contacts=past_contacts_array[i],
                num_time_steps=num_time_steps)

            d_noterm_cumsum = np.cumsum(d_no_term)

            num_days_transit = num_days_s - 1
            num_days_transit[num_days_transit < 0] = 0

            d_penalties = (
                np.take(d_noterm_cumsum, num_days_transit)
                + np.take(d_term, num_days_s))

            # Calculate log_joint
            # Numba only does matmul with 2D-arrays, so do reshaping below
            # log_c and log_a are described in the CRISP paper (Herbrich et al. 2021)
            log_joint = log_c_z_u[i] + log_A_start + d_penalties

            joint_distr = softmax(log_joint).astype(np.single)

            # Calculate the posterior expectations, with complete enumeration
            post_exps[i] = np.reshape(np.dot(
                seq_array_hot.reshape(num_time_steps*4, num_sequences), joint_distr),
                (num_time_steps, 4))

    with numba.objmode(t1='f8'):
        t1 = time.time()
    # t0, t1 = 0,0
    return post_exps, t0, t1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fact_neigh(
    num_users: int,
    app_user_ids: np.ndarray,
    app_users: np.ndarray,
    non_app_user_ids: np.ndarray,
    num_time_steps: int,
    observations_all: np.ndarray,
    contacts_all: np.ndarray,
    probab_0: float,
    probab_1: float,
    g_param: float,
    h_param: float,
    alpha: float,
    beta: float,
    infection_prior: float,
    linear_feature_imputation: Any,
    neural_feature_imputation: Any,
    infection_rate: float,
    prev_z_states: np.ndarray,
    mse_states: np.ndarray,
    local_mean_baseline: bool,
    clip_lower: float = -1.,
    clip_upper: float = 10000.,
    quantization: int = -1,
    users_stale: Optional[np.ndarray] = None,
    num_updates: int = 5,
    verbose: bool = False,
    trace_dir: Optional[str] = None,
    diagnostic: Optional[Any] = None,
) -> np.ndarray:
    """Inferes latent states using Factorised Neighbor method.

    Uses Factorised Neighbor approach from
    'The dlr hierarchy of approximate inference, Rosen-Zvi, Jordan, Yuille, 2012'

    Update equations described in
    'No time to waste: ..., Romijnders et al. AISTATS 2023

    Args:
      num_users: Number of users to infer latent states
      num_time_steps: Number of time steps to infer latent states
      observations_all: List of all observations
      contacts_all: List of all contacts
      probab_0: Probability to be infected spontaneously
      probab_1: Probability of transmission given contact
      g_param: float, dynamics parameter, p(E->I|E)
      h_param: float, dynamics parameter, p(I->R|I)
      alpha: False positive rate of observations, (1 minus specificity)
      beta: False negative rate of observations, (1 minus sensitivity)
      quantization: number of levels for quantization. Negative number indicates
        no use of quantization.
      num_updates: Number of rounds to update using Factorised Neighbor algorithm
      verbose: set to true to get more verbose output

    Returns:
      array in [num_users, num_timesteps, 4] being probability of over
      health states {S, E, I, R} for each user at each time step
    """
    del diagnostic
    if users_stale is not None:
        raise NotImplementedError("Stale users not implemented")
    t_start_preamble = time.time()

    assert clip_lower < 1
    assert clip_upper > 0

    mse_total = -1
    mae_total = -1
    # Seq Array is the w in CRISP paper which includes (t_0, d_E, d_I)
    seq_array = np.stack(list(
        # TODO: change start_se to True
        util.iter_sequences(time_total=num_time_steps, start_se=False)))
    seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
        seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int32)

    # log_c and log_a are described in the CRISP paper (Herbrich et al. 2021)
    # Log_a start returns an array of values for each seq_array option
    log_A_start = util.enumerate_log_prior_values(
        [1-probab_0, probab_0, 0., 0.], [1-probab_0, 1-g_param, 1-h_param],
        seq_array, num_time_steps)

    # obs_array[t, :, i] is about the log-likelihood of the observation being
    # i=0 or i=1, at time step t.
    # These values then are timestep dependent and dependent on which state the user
    # is in. We map the obs_array value to a specific user and make predictions
    # based on that.
    obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)

    # Precompute log(C) terms, relating to observations
    # Log_c_z_u has test
    log_c_z_u = util.calc_c_z_u(
        (0, num_users),
        obs_array,
        observations_all)

    q_marginal_infected = np.zeros(
        (num_users, num_time_steps), dtype=np.single)

    post_exp = np.zeros((num_users, num_time_steps, 4), dtype=np.single)

    t_preamble1 = time.time() - t_start_preamble
    t_start_preamble = time.time()

    num_max_msg = constants.CTC
    past_contacts, max_num_contacts = util.get_past_contacts_static(
        (0, num_users), contacts_all, num_msg=num_max_msg, app_users=app_users)

    if max_num_contacts >= num_max_msg:
        logger.warning(
            f"Max number of contacts {max_num_contacts} >= {num_max_msg}")

    if neural_feature_imputation:
        mse_total = 0
        mae_total = 0
        model = neural_feature_imputation['model'].to(device).eval()
        onehot_encodings = neural_feature_imputation['onehot_encodings']
        logger.info('Neural imputation starting')
        t_n_imputation_1 = time.time()
        infection_prior = np.float64(infection_prior)

        X_full, positions, imputation_masks, user_ids = util.impute_neural_data_collection(
            app_user_ids,
            past_contacts,
            infection_prior,
            infection_rate,
            onehot_encodings,
            prev_z_states,
        )

        user_ids = np.array(user_ids, dtype=np.int32)
        logger.info(
            f"Time spent on data collection {time.time() - t_n_imputation_1} seconds")

        X_tensor = torch.tensor(np.array(X_full), dtype=torch.float32)
        dataset = Dset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=16384, shuffle=False)

        t_n_imputation_1 = time.time()

        preds = util.neural_imp_predictions(
            model,
            dataloader
        )

        logger.info(
            f"Time spent on neural imputation preds {time.time() - t_n_imputation_1} seconds")
        do_mse = not np.all(mse_states == -1.)

        t_n_imputation_1 = time.time()

        mse_total, mae_total = util.neural_imp_fill_contacts(
            user_ids,
            past_contacts,
            positions,
            imputation_masks,
            preds,
            mse_states,
            do_mse,
        )

        logger.info(
            f"Time spent on neural imputation fill {time.time() - t_n_imputation_1} seconds")

        infection_prior = -1.

    if linear_feature_imputation:
        mse_total = 0
        mae_total = 0
        # Modify past contacts in place with feature imputation model.
        weights = linear_feature_imputation['weights']
        intercept = linear_feature_imputation['intercept']
        onehot_encodings = linear_feature_imputation['onehot_encodings']
        infection_prior = np.float64(infection_prior)
        t_linreg_1 = time.time()
        do_mse = not np.all(mse_states == -1.)

        mse_total, mae_total = util.impute_lin_reg(
            app_user_ids,
            past_contacts,
            infection_prior,
            infection_rate,
            weights,
            intercept,
            onehot_encodings,
            prev_z_states,
            mse_states,
            do_mse
        )

        logger.info(
            f"Time spent on linear regression {time.time() - t_linreg_1} seconds")
        infection_prior = -1.

    if local_mean_baseline:
        mse_total = 0
        mae_total = 0
        do_mse = not np.all(mse_states == -1.)
        # In place modification of past contacts based on local contact graphs.

        mse_total, mae_total = util.impute_local_graph(
            prev_z_states,
            app_user_ids,
            past_contacts,
            mse_states,
            do_mse
        )

        infection_prior = -1.

    t_preamble2 = time.time() - t_start_preamble

    logger.info(
        f"Time spent on preamble1/preamble2 {t_preamble1:.1f}/{t_preamble2:.1f}")
    for num_update in range(num_updates):
        if verbose:
            logger.info(f"Num update {num_update}")

        post_exp, tstart, t_end = fn_step_wrapped(
            (0, num_users),
            seq_array_hot,
            log_c_z_u,
            log_A_start,
            q_marginal_infected,
            num_time_steps,
            probab_0,
            probab_1,
            clip_lower=-1.,
            clip_upper=10000.,
            past_contacts_array=past_contacts,
            quantization=quantization,
            infection_prior=infection_prior,
            app_users=app_users,
            non_app_user_ids=non_app_user_ids,
            prev_z_states=prev_z_states,
        )

        if verbose:
            logger.info(f"Time for fn_step: {t_end - tstart:.1f} seconds")

        q_marginal_infected[app_user_ids, :] = post_exp[app_user_ids, :, 2]

    post_exp_collect = post_exp

    return post_exp_collect, {'mse': mse_total, 'mae': mae_total}
