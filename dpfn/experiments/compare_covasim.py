import covasim as cv
import numpy as np
import os
import psutil
import threading
import time
import torch
from typing import Any, Dict, Optional, Union
import warnings
from dpfn.experiments import (
    prequential, util_experiments, prequential, util_covasim, util_dataset)
from dpfn import logger
from joblib import load
from experiments.model_utils import make_predictions
from experiments.util_dataset import create_dataset
from dpfn.util import get_onehot_encodings, bootstrap_sampling_ave_precision


def compare_policy_covasim(
    inference_method: str,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    arg_rng: int,
    neural_imp_model: Any,
    trace_dir: Optional[str] = None,
    trace_dir_preds: Optional[str] = None,
    do_diagnosis: bool = False,
    modify_contacts: bool = False,
    run_mean_baseline: bool = False,
    run_local_mean_baseline: bool = False,
    dl_model=None
):
    """Compares different inference algorithms on the supplied contact graph."""
    del do_diagnosis

    # Start a daemon to log to wandb
    daemon_wandb = threading.Thread(
        target=util_covasim.log_to_wandb, args=(runner,), daemon=True)
    daemon_wandb.start()

    num_time_steps = cfg["data"]["num_time_steps"]
    num_users = cfg["data"]["num_users"]
    # Daily fraction of the population that gets a test
    fraction_test = cfg["data"]["fraction_test"]
    # Probability of the person being lost-to-follow-up after a test
    loss_prob = cfg["data"]["loss_prob"]
    t_start_quarantine = cfg["data"]["t_start_quarantine"]

    std_rank_noise = cfg['std_rank_noise']

    num_days_window = cfg["model"]["num_days_window"]
    quantization = cfg["model"]["quantization"]
    num_rounds = cfg["model"]["num_rounds"]
    pred_days = cfg['model']['pred_days']

    model_type = cfg['dl_model_type']
    add_weights = (model_type == 'gcn_weight' or model_type == 'gcn_global')
    feature_prop = cfg['feature_propagation']
    feature_imp_model = None
    one_hot_encoder = None

    neural_feature_imputation = neural_imp_model

    linear_feature_imputation = {}
    if cfg.get('feature_imp_model'):
        feature_imp_model, one_hot_encoder = load(
            'dpfn/config/feature_imp_configs/' + cfg.get('feature_imp_model'))
        possible_values = [0, 1, 2, 3]
        one_hot_encodings = get_onehot_encodings(
            possible_values, one_hot_encoder)
        linear_feature_imputation['weights'] = np.array(
            feature_imp_model.coef_)
        linear_feature_imputation['intercept'] = np.float64(
            feature_imp_model.intercept_[0])
        linear_feature_imputation['onehot_encodings'] = one_hot_encodings

    neural_feature_imputation = {}
    if neural_imp_model:
        possible_values = [0, 1, 2, 3]
        one_hot_encodings = get_onehot_encodings(
            possible_values, neural_imp_model['one_hot_encoder'])

        neural_feature_imputation['model'] = neural_imp_model['model']
        neural_feature_imputation['onehot_encodings'] = one_hot_encodings

    # Percentage of app users in population
    app_users_fraction = cfg["data"]["app_users_fraction"]

    if 'app_users_fraction_wandb' in cfg:
        app_users_fraction = cfg.get("app_users_fraction_wandb", -1)

    logger.info(f"App users fraction: {app_users_fraction}")
    logger.info(f"STD_rank_noise: {cfg['std_rank_noise']}")
    assert app_users_fraction >= 0 and app_users_fraction <= 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if feature_imp_model:
        logger.info('Running with linear regression feature imputation')
    elif run_mean_baseline:
        logger.info('Running mean baseline')
    elif run_local_mean_baseline:
        logger.info('Running local mean baseline')
    else:
        logger.info('Running vanilla factorized neighbors')

    seed = cfg.get("seed", 123)

    online_mse = cfg.get('online_mse')
    mse_inference_func = None
    online_mse_mse_states_placeholder = -1. * \
        np.ones((1, 1, 4), dtype=np.float32)
    if online_mse:
        mse_inference_func, _ = util_experiments.make_inference_func(
            'fn', num_users, cfg, trace_dir=None)

    logger.info((
        f"Settings at experiment: {quantization:.0f} quant, at {fraction_test}% "
        f"seed {seed}"))

    sensitivity = 1. - cfg["data"]["alpha"]

    if cfg["model"]["beta"] == 0:
        logger.warning(
            "COVASIM does not model false positives yet, setting to 0")

    cfg["model"]["beta"] = 0
    cfg["data"]["beta"] = 0
    # assert num_time_steps == 91, "hardcoded 91 days for now, TODO: fix this"

    t0 = time.time()

    # Make simulator
    pop_infected = 25
    if num_users >= 100000:
        pop_infected = 50
    if num_users > 500000:
        pop_infected = 100

    pars = {
        "pop_type": 'hybrid',
        "pop_size": num_users,
        "pop_infected": pop_infected,
        "start_day": '2020-02-01',
        "end_day": '2020-05-01',
    }

    inference_func, do_random = util_experiments.make_inference_func(
        inference_method, num_users, cfg, trace_dir=trace_dir)

    def subtarget_func_random(sim, history):
        """Subtarget function for random testing.

        This function is run every day after the contacts are sampled and before
        the tests are run. It returns a dictionary with the keys 'inds' and 'vals'.
        """
        inds = sim.people.uid
        # vals = np.random.rand(len(sim.people))  # Create the array

        # Uncomment these lines to deterministically test people that are exposed
        app_users = sim.app_users
        app_user_ids = np.nonzero(app_users)[0]

        vals = np.zeros(len(sim.people))  # Create the array
        exposed = cv.true(sim.people.exposed)
        infected = cv.true(sim.people.infectious)
        users_of_interest = np.concatenate((exposed, infected))

        exposed_appusers_intersect = np.intersect1d(
            users_of_interest, app_user_ids)
        vals[exposed_appusers_intersect] = 100  # Probability for testing

        states_today = 3*np.ones(num_users, dtype=np.int32)
        states_today[sim.people.exposed] = 2
        states_today[sim.people.infectious] = 1
        states_today[sim.people.susceptible] = 0

        app_user_preds = vals[app_user_ids]
        app_user_states = states_today[app_user_ids]
        app_user_states = np.where(
            (app_user_states == 2) | (app_user_states == 1), 1, 0)
        auroc = bootstrap_sampling_ave_precision(
            app_user_preds, app_user_states)

        history['aurocs'][sim.t] = auroc
        return {'inds': inds, 'vals': vals}, history

    def subtarget_func_inference(sim, history):
        """Subtarget function for testing.

        This function is run every day after the contacts are sampled and before
        the tests are run. It returns a dictionary with the keys 'inds' and 'vals'.
        """
        assert isinstance(history, dict)

        users_age = sim.users_age
        app_users = sim.app_users

        pred_placeholder = np.zeros((num_users), dtype=np.float32)
        state_preds = np.zeros((num_users), dtype=np.float32)

        app_user_ids = np.nonzero(app_users)[0]
        non_app_user_ids = np.where(app_users == 0)[0]

        # For imputation measurements and how accurate the system is comparing to 100% FN
        mse_at_t_imp = 0
        mae_at_t_imp = 0
        mse_at_t_NA = 0
        mae_at_t_NA = 0

        user_age_pinf_mean = -1. * np.ones((9), dtype=np.float32)
        infection_prior = -1.

        if sim.t > t_start_quarantine:
            contacts = sim.people.contacts

            layer_dict = {
                'h': 0,
                's': 1,
                'w': 2,
                'c': 3
            }

            contacts_add = []
            for layerkey in contacts.keys():
                layer_code = layer_dict[layerkey]
                ones_vec = np.ones_like(contacts[layerkey]['p1'])
                layer_vec = layer_code * ones_vec
                contacts_add.append(np.stack((
                    contacts[layerkey]['p1'],
                    contacts[layerkey]['p2'],
                    sim.t*ones_vec,
                    layer_vec,
                ), axis=1))
                contacts_add.append(np.stack((
                    contacts[layerkey]['p2'],
                    contacts[layerkey]['p1'],
                    sim.t*ones_vec,
                    layer_vec,
                ), axis=1))

            contacts_today = np.concatenate(contacts_add, axis=0)

            history['contacts'] = np.concatenate(
                (history['contacts'], contacts_today), axis=0)

            # TODO make this more efficient
            num_days = min((num_days_window, sim.t))
            history['contacts'] = history['contacts'][
                history['contacts'][:, 2] > sim.t - num_days]
            history['observations'] = history['observations'][
                history['observations'][:, 1] > sim.t - num_days]

            contacts_rel = np.copy(history['contacts']).astype(np.int32)
            obs_rel = np.copy(history['observations']).astype(np.int32)

            contacts_rel[:, 2] = contacts_rel[:, 2] - sim.t + num_days - 1
            obs_rel[:, 1] = obs_rel[:, 1] - sim.t + num_days - 1

            # TODO remove this line. Follow up with github issue
            # https://github.com/InstituteforDiseaseModeling/covasim/issues/400
            is_not_double = np.logical_not(
                contacts_rel[:, 0] == contacts_rel[:, 1])
            contacts_rel = contacts_rel[is_not_double]

            # assert 0 <= contacts_rel[:, 2].min() <= sim.t, (
            #   f"Earliest contact {contacts_rel[:, 2].min()} is before {sim.t}")
            # assert 0 <= obs_rel[:, 1].min() <= sim.t, (
            #   f"Earliest obs {obs_rel[:, 1].min()} is before {sim.t}")

            if run_mean_baseline:
                infection_prior = history['infection_prior'][sim.t - 1]

                if online_mse:
                    mse_prior = infection_prior * \
                        np.ones((len(non_app_user_ids)), dtype=np.float32)
                    mse_at_t_imp = (
                        (history['infection_state_mse'][non_app_user_ids, sim.t - 1, 2] - mse_prior) ** 2).mean()
                    mae_at_t_imp = (np.absolute(
                        history['infection_state_mse'][non_app_user_ids, sim.t - 1, 2] - mse_prior)).mean()

            if history['infection_state'][sim.t - 1].any() > 0:
                pred_placeholder = history['infection_state'][sim.t - 1]
            # Add +1 so the model predicts one day into the future

            analysis = sim.get_analyzer('analysis')
            infection_rates = analysis.e_rate + analysis.i_rate

            t_start = time.time()
            pred, contacts_age, mse_loss = inference_func(
                observations_list=obs_rel,
                contacts_list=contacts_rel,
                app_user_ids=app_user_ids,
                app_users=app_users,
                non_app_user_ids=non_app_user_ids,
                num_updates=num_rounds,
                num_time_steps=num_days + 1,
                infection_prior=infection_prior,
                linear_feature_imputation=linear_feature_imputation,
                neural_feature_imputation=neural_feature_imputation,
                infection_rate=np.float64(infection_rates[sim.t - 1]),
                local_mean_baseline=run_local_mean_baseline,
                prev_z_states=pred_placeholder,
                mse_states=history['infection_state_mse'][:,
                                                          :sim.t] if online_mse else online_mse_mse_states_placeholder,
            )
            pred_dump = np.copy(pred)
            pred[app_users == 0] = np.zeros((4), dtype=np.float32)

            if mse_loss['mae'] >= 0:
                mae_at_t_imp = mse_loss['mae']
                mse_at_t_imp = mse_loss['mse']

            history['mse_values_imputation'][sim.t] = mse_at_t_imp
            history['mae_values_imputation'][sim.t] = mae_at_t_imp

            if online_mse:
                pred_mse, _, _ = mse_inference_func(
                    obs_rel,
                    contacts_rel,
                    app_user_ids,
                    app_users,
                    non_app_user_ids,
                    num_rounds,
                    num_time_steps=num_days + 1,
                    infection_prior=-1.,
                    user_age_pinf_mean=user_age_pinf_mean,
                    linear_feature_imputation=None,
                    neural_feature_imputation=None,
                    infection_rate=np.float64(infection_rates[sim.t - 1]),
                    local_mean_baseline=False,
                    prev_z_states=None,
                    mse_states=None,
                )

            if dl_model:
                logger.info('Deep learning predictions')

                user_free = np.logical_not(sim.people.isolated)
                incorporated_users = app_users & user_free

                logger.info('Measuring time for model_data creation')
                start_time = time.time()

                model_data = util_dataset.inplace_features_data_creation(
                    contacts_rel, obs_rel, pred, user_free,
                    users_age, app_users, num_users, num_time_steps, app_user_ids, infection_rates[
                        sim.t - 1],
                )
                end_time = time.time()
                logger.info(
                    f"Time taken for model_data creation: {end_time - start_time} seconds")

                logger.info('Measuring time for create_dataset')

                start_time = time.time()
                if run_mean_baseline:
                    # infection_prior_now = np.mean(pred[app_user_ids, -1, 2])
                    train_loader, dataset_user_ids = create_dataset(
                        model_data, model_type, cfg, infection_prior=infection_prior, add_weights=add_weights)
                else:
                    train_loader, dataset_user_ids = create_dataset(
                        model_data, model_type, cfg, add_weights=add_weights, local_mean_base=run_local_mean_baseline)

                end_time = time.time()
                logger.info(
                    f"Time taken for dataset creation: {end_time - start_time} seconds")

                all_preds = []
                all_preds = make_predictions(
                    dl_model,
                    train_loader,
                    model_type,
                    device,
                    feature_prop=feature_prop
                )

                # Reset statistics, since the incorporated users can change.
                if np.all(all_preds == 0.0):
                    logger.info('All predictions zero.')

                state_preds = np.zeros((num_users), dtype=np.float32)
                state_preds[dataset_user_ids] = all_preds

                if trace_dir_preds is not None:
                    logger.info('Dumping prediction values')
                    util_dataset.dump_preds(
                        pred[:, -1, 2], state_preds, incorporated_users, sim.t, trace_dir_preds, app_user_ids, users_age)

            rank_score = pred[:, -1, 1] + pred[:, -1, 2] + state_preds
            # rank_score = pred[:, -1, 2] + state_preds
            time_spent = time.time() - t_start

            if online_mse:
                mse_at_t_NA = (
                    (pred_mse[app_user_ids, -1, 2] - rank_score[app_user_ids])**2).mean()
                mae_at_t_NA = (np.absolute(
                    pred_mse[app_user_ids, -1, 2] - rank_score[app_user_ids])).mean()

                history['mse_values_NA'][sim.t] = mse_at_t_NA
                history['mae_values_NA'][sim.t] = mae_at_t_NA

            if std_rank_noise > 0:
                rank_score += std_rank_noise * np.random.randn(num_users)

            # Track some metrics here:
            # Exposed is superset of infectious, but this is overwritten below
            states_today = 3*np.ones(num_users, dtype=np.int32)
            states_today[sim.people.exposed] = 2
            states_today[sim.people.infectious] = 1
            states_today[sim.people.susceptible] = 0

            history['infection_prior'][sim.t] = np.mean(
                pred[app_user_ids, -1, 2])

            app_user_preds = rank_score[app_user_ids]
            app_user_states = states_today[app_user_ids]
            app_user_states = np.where(
                (app_user_states == 2) | (app_user_states == 1), 1, 0)
            auroc = bootstrap_sampling_ave_precision(
                app_user_preds, app_user_states)

            if trace_dir is not None and sim.t > 10:
                user_free = np.logical_not(sim.people.isolated)
                analysis = sim.get_analyzer('analysis')
                infection_rates = analysis.e_rate + analysis.i_rate

                rate_user_free = np.mean(user_free)
                rate_infection = np.mean(np.logical_or(
                    states_today == 1, states_today == 2))
                rate_diagnosed = np.mean(sim.people.diagnosed)
                logger.info((
                    f"Day {sim.t}: {rate_user_free:.4f} free, "
                    f"{rate_infection:.4f} infection, {rate_diagnosed:.4f} diagnosed"))

                util_dataset.dump_features_graph(
                    contacts_now=contacts_rel,
                    observations_now=obs_rel,
                    z_states_inferred=pred_dump,
                    user_free=user_free,
                    z_states_sim=states_today,
                    users_age=users_age,
                    app_users=app_users,
                    trace_dir=trace_dir,
                    num_users=num_users,
                    num_time_steps=num_time_steps,
                    t_now=sim.t,
                    rng_seed=int(seed),
                    infection_prior=infection_prior,
                    infection_prior_now=history['infection_prior'][sim.t],
                    infection_rate_prev=infection_rates[sim.t - 1],)

            p_at_state = pred[range(num_users), -1, states_today]
            history['likelihoods_state'][sim.t] = np.mean(
                np.log(p_at_state+1E-9))
            history['ave_prob_inf_at_inf'][sim.t] = np.mean(
                p_at_state[states_today == 2])
            history['time_inf_func'][sim.t] = time_spent
            history['infection_state'][sim.t] = pred[:, -1, 2]
            history['aurocs'][sim.t] = auroc
            if online_mse:
                history['infection_state_mse'][:, sim.t] = pred_mse[:, -1]

        else:
            # For the first few days of a simulation, just test randomly
            rank_score = np.ones(num_users) + np.random.rand(num_users)

        output = {'inds': sim.people.uid, 'vals': rank_score}
        return output, history

    # TODO: fix this to new keyword
    subtarget_func = (
        subtarget_func_random if do_random else subtarget_func_inference)

    # TODO: test fraction should be app_users_frac_num if its lower, since we can
    # only test the app users.
    test_intervention = cv.test_num(
        daily_tests=int(fraction_test*num_users),
        num_users=num_users,
        do_plot=False,
        sensitivity=sensitivity,  # 1 - false_negative_rate
        loss_prob=loss_prob,  # probability of the person being lost-to-follow-up
        subtarget=subtarget_func,
        label='intervention_history')

    # Create, run, and plot the simulations
    logger.info(f'Num time steps: {num_time_steps}')
    sim = cv.Sim(
        pars,
        interventions=test_intervention,
        analyzers=util_covasim.StoreSEIR(
            num_days=num_time_steps, label='analysis'),
        app_users_fraction=app_users_fraction)

    # COVASIM run() runs the entire simulation, including the initialization
    sim.set_seed(seed=seed)

    sim.run(reset_seed=True)

    analysis = sim.get_analyzer('analysis')
    history_intv = sim.get_intervention('intervention_history').history
    logger.info(analysis.e_rate)
    logger.info(analysis.i_rate)

    infection_rates = analysis.e_rate + analysis.i_rate
    user_infection_rates = analysis.user_e_rate + analysis.user_i_rate
    peak_crit_rate = np.max(analysis.crit_rate)
    peak_user_crit_rate = np.max(analysis.user_crit_rate)

    # Calculate PIR and Drate
    time_pir, pir = np.argmax(infection_rates), np.max(infection_rates)
    user_pir = np.max(user_infection_rates)

    app_users = sim.app_users

    total_drate = sim.people.dead.sum() / len(sim.people)
    user_total_drate = sim.people.dead[app_users].sum() / app_users.sum()

    prequential.dump_results_json(
        datadir=results_dir,
        cfg=cfg,
        precisions=analysis.precisions.tolist(),
        recalls=analysis.recalls.tolist(),
        user_precisions=analysis.user_precisions.tolist(),
        user_recalls=analysis.user_recalls.tolist(),
        exposed_rates=analysis.e_rate.tolist(),
        infection_rates=infection_rates.tolist(),
        user_infection_rates=user_infection_rates.tolist(),
        num_quarantined=analysis.isolation_rate.tolist(),
        critical_rates=analysis.crit_rate.tolist(),
        likelihoods_state=history_intv['likelihoods_state'].tolist(),
        ave_prob_inf_at_inf=history_intv['ave_prob_inf_at_inf'].tolist(),
        mae_avg_imp=float(history_intv['mae_values_imputation'].mean()),
        mse_avg_imp=float(history_intv['mse_values_imputation'].mean()),
        mae_avg_na=float(history_intv['mae_values_NA'].mean()),
        mse_avg_na=float(history_intv['mse_values_NA'].mean()),
        avg_auroc=float(history_intv['aurocs'].mean()),
        inference_method=inference_method,
        name=runner.name,
        pir=float(np.max(infection_rates)),
        pcr=float(np.max(analysis.crit_rate)),
        total_drate=float(total_drate),
        user_total_drate=float(user_total_drate),
        quantization=quantization,
        seed=cfg.get("seed", -1))

    logger.info((
        f"At day {time_pir} peak infection rate is {pir:.5f} "
        f"and total death rate is {total_drate:.5f}"))

    _, loadavg5, loadavg15 = os.getloadavg()
    swap_use = psutil.swap_memory().used / (1024.0 ** 3)

    time_spent = time.time() - t0
    logger.info(f"With {num_rounds} rounds, PIR {pir:5.2f}")
    results = {
        "time_spent": time_spent,
        "pir_mean": pir,
        "user_pir_mean": user_pir,
        "pcr": peak_crit_rate,
        "user_pcr": peak_user_crit_rate,
        "total_drate": total_drate,
        "user_drate": user_total_drate,
        "loadavg5": loadavg5,
        "loadavg15": loadavg15,
        "swap_use": swap_use,
        "mse_avg_na": float(history_intv['mse_values_NA'].mean()),
        "mse_avg_imp": float(history_intv['mse_values_imputation'].mean()),
        "avg_auroc": float(history_intv['aurocs'].mean()),
        "recall": np.nanmean(analysis.recalls[10:]),
        "precision": np.nanmean(analysis.precisions[10:]),
        "user_recall": np.nanmean(analysis.user_recalls[10:]),
        "user_precision": np.nanmean(analysis.user_precisions[10:]),
    }
    runner.log(results)

    return results
