import argparse
import numpy as np
import numba
import os
import random
import socket
import traceback
import wandb
import json
from collections import defaultdict
from config import config
from experiments import (util_experiments)
from dpfn import LOGGER_FILENAME, logger
from dpfn import util
from dpfn import util_wandb
from experiments.compare_covasim import compare_policy_covasim
from experiments.compare_abm import compare_abm
from experiments.model_utils import get_model, get_neural_imp_model
import torch
from constants import GRAPH_MODELS
from joblib import load


if __name__ == "__main__":
    all_model_types = np.concatenate((GRAPH_MODELS, ['set']))
    
    parser = argparse.ArgumentParser(
        description='Compare statistics acrosss inference methods')
    parser.add_argument('--inference_method', type=str, default='fn',
                        choices=[
                            'fn', 'dummy', 'random', 'dpct', 'fncpp'],
                        help='Name of the inference method')
    parser.add_argument('--simulator', type=str, default='abm',
                        choices=['abm', 'covasim'],
                        help='Name of the simulator')
    parser.add_argument('--config_data', type=str, default='intermediate_graph_abm_02',
                        help='Name of the config file for the data')
    parser.add_argument('--config_model', type=str, default='model_ABM01',
                        help='Name of the config file for the model')
    parser.add_argument('--name', type=str, default=None,
                        help=('Name of the experiments. WandB will set a random'
                              ' when left undefined'))
    parser.add_argument('--do_diagnosis', action='store_true')
    parser.add_argument('--dump_traces', action='store_true')
    parser.add_argument('--app_users_fraction', type=float, default=None)
    parser.add_argument('--modify_contacts', action='store_true')
    parser.add_argument('--age_baseline', action='store_true')
    parser.add_argument('--mean_baseline', action='store_true', help='Global average feature imputation baseline')
    parser.add_argument('--local_mean_baseline', action='store_true')
    parser.add_argument('--gaussian_baseline', action='store_true', help='Gaussian sampling for feature imputation')
    parser.add_argument('--static_baseline', action='store_true')
    parser.add_argument('--static_baseline_path', type=str,
                        default='dpfn/data/static_mean_baseline_results')
    parser.add_argument('--seed_value', type=int, default=None)
    parser.add_argument('--model',
                        type=str,
                        help="Type of deep learning model to apply to FN",
                        default=None,
                        choices=all_model_types)
    parser.add_argument('--model_name',
                        type=str,
                        default=None,
                        help='The model state dict that will be used to load the model')
    parser.add_argument('--n_layers',
                        type=int,
                        default=1,
                        help='Number of layers in the deep learning model.')
    parser.add_argument('--nhid',
                        type=int,
                        default=256,
                        help='Number of hidden dimensions in the deep learning model.')
    parser.add_argument('--num_users',
                        type=int,
                        default=None)
    parser.add_argument('--num_time_steps',
                        type=int,
                        default=None),
    parser.add_argument('--std_rank_noise',
                        type=float,
                        default=0,
                        help='Noise added to testing rankings to generate more positive samples, hence more data')
    parser.add_argument('--dump_traces_folder', 
                        type=str,
                        default='datadump_mean',
                        help='Folder name where to dump traces')
    parser.add_argument('--collect_pred_data',
                        action='store_true',
                        help='Saves the differences in FN and DL predictions in different formats for explainability.')
    parser.add_argument('--feature_propagation',
                        action='store_true',
                        help='Whether to do feature propagation or not.')
    parser.add_argument('--feature_imp_model',
                        type=str,
                        default=None,
                        help='Path to the feature imputation model')
    parser.add_argument('--online_mse', action='store_true')
    parser.add_argument('--one_hot_encoding',
                        action='store_true',
                        help='Whether interaction type should be converted into one hot or not.')
    parser.add_argument('--neural_imputation_model_path',
                        type=str,
                        default=None,
                        help='Path to the neural imputation model, when given it is run.')
    parser.add_argument('--testing_fraction',
                        type=float,
                        default=None,)

    num_threads = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    numba.set_num_threads(num_threads)
    logger.info(f'Running on device: {device}')
    logger.info(f"SLURM env N_TASKS: {os.getenv('SLURM_NTASKS')}")
    logger.info(f"Start with {num_threads} threads")

    args = parser.parse_args()
    # Both baselines should not be used at the same time.
    assert sum([args.age_baseline, args.mean_baseline]) <= 1
 
    configname_data = args.config_data
    configname_model = args.config_model
    fname_config_data = f"dpfn/config/{configname_data}.ini"
    fname_config_model = f"dpfn/config/{configname_model}.ini"
    data_dir = f"dpfn/data/{configname_data}/"
    inf_method = args.inference_method
    
    config_data = config.ConfigBase(fname_config_data)
    config_model = config.ConfigBase(fname_config_model)

    # Start WandB
    config_wandb = defaultdict(None)
    config_wandb['config_data_name'] = configname_data
    config_wandb['config_model_name'] = configname_model
    config_wandb['dl_model_name'] = args.model_name
    config_wandb['dl_model_type'] = args.model
    config_wandb['cpu_count'] = util.get_cpu_count()
        
    config_wandb['data'] = config_data.to_dict()
    config_wandb['model'] = config_model.to_dict()
    
    if args.testing_fraction is not None:
        config_wandb["data"]["fraction_test"] = args.testing_fraction
        
    config_wandb['std_rank_noise'] = args.std_rank_noise
    config_wandb['feature_propagation'] = args.feature_propagation
    config_wandb['feature_imp_model'] = args.feature_imp_model
    config_wandb['online_mse'] = args.online_mse
    config_wandb['one_hot'] = args.one_hot_encoding
    config_wandb['simulator'] = args.simulator
    config_wandb['gaussian_baseline'] = args.gaussian_baseline

    if args.num_time_steps:
        config_wandb["data"]["num_time_steps"] = args.num_time_steps

    if args.num_users:
        config_wandb['data']['num_users'] = args.num_users

    if args.app_users_fraction:
        config_wandb["data"]["app_users_fraction"] = float(
            args.app_users_fraction)
        
    neural_imp_model = None
    if args.neural_imputation_model_path:
        neural_imp_model = {}
        neural_imp_model['model'] = get_neural_imp_model(args.neural_imputation_model_path, args.simulator, device)
        neural_imp_model['one_hot_encoder'] = neural_one_hot_encoder = load('dpfn/config/feature_imp_configs/' + args.neural_imputation_model_path.split('.')[0] + '.joblib')

    # WandB tags
    tags = [
        str(args.simulator), inf_method, f"cpu{util.get_cpu_count()}",
        configname_data, configname_model]
    tags.append("dump_traces" if args.dump_traces else "nodump")

    tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")

    do_wandb = 'int' not in socket.gethostname()
    if do_wandb:
        runner_global = wandb.init(
            project="dpfn",
            notes=" ",
            name=args.name,
            tags=tags,
            config=config_wandb,
        )
        config_wandb = config.clean_hierarchy(dict(runner_global.config))
    else:
        runner_global = util_wandb.WandbDummy()
    config_wandb = util_experiments.set_noisy_test_params(config_wandb)
    config_wandb = util_experiments.convert_log_params(config_wandb)
    logger.info(config_wandb)
    # Prepare model if model given
    
    # Set random seed
    seed_value = config_wandb.get("seed", -1)
    if seed_value < 0:
        if args.seed_value:
            seed_value = args.seed_value
        else:
            seed_value = random.randint(0, 999)
            
    logger.info(f"Seed value: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    
    if args.model:
        logger.info(f'Running with the deep model: {str(args.model)}')
        num_features = 5
        if args.simulator == 'covasim' and args.one_hot_encoding:
            num_features = 8
        elif args.simulator == 'abm' and args.one_hot_encoding:
            num_features = 7
            
        dl_model = get_model(args.model, n_layers=args.n_layers, nhid=args.nhid, num_features=num_features).to(device)
        saved_model = torch.load(f"dpfn/config/dl_configs/" + args.simulator + '/' + args.model_name, map_location=torch.device(device))
        dl_model.load_state_dict(saved_model)

        if args.model == 'gcn_weight':
            logger.info(f"Three of the model weights: {dl_model.msg_weights}")
        dl_model.eval()

    else:
        dl_model = None
    
    # Random number generator to pass as argument to some imported functions
    arg_rng = np.random.default_rng(seed=seed_value)

    # Set up locations to store results
    if args.simulator == 'covasim':
        sim_name = 'cov'
    else:
        sim_name = 'abm'
    if args.model:
      input_str = args.name
      experiment_name = f'{input_str}_run_{sim_name}_seed_model_' + \
          str(args.model) + '_seed_' + str(seed_value)
    elif args.inference_method == 'random':
        experiment_name = f'run_{sim_name}_oracle_seed_' + str(seed_value)
    elif args.static_baseline:
        if args.mean_baseline:
            experiment_name = f'run_{sim_name}_static_mean_seed_' + str(seed_value)
        elif args.age_baseline:
            experiment_name = f'run_{sim_name}_static_age_seed_' + str(seed_value)
    elif args.feature_imp_model:
        experiment_name = f'run_{sim_name}_linreg_seed_' + str(seed_value)
    elif args.neural_imputation_model_path:
        experiment_name = f'run_{sim_name}_neural_imp_seed_' + str(seed_value)
    elif args.mean_baseline:
      experiment_name = f'run_{sim_name}_mean_seed_' + str(seed_value)
    elif args.local_mean_baseline:
      experiment_name = f'run_{sim_name}_local_mean_seed_' + str(seed_value)
    elif args.age_baseline:
      experiment_name = f'run_{sim_name}_age_seed_' + str(seed_value)
    else:
      experiment_name = f'run_{sim_name}_seed_' + str(seed_value)

    if args.dump_traces:
      experiment_name = experiment_name + '_dump_traces'

    if args.app_users_fraction:
        experiment_name = experiment_name + \
            "_adaption_" + str(args.app_users_fraction)
          
    elif 'app_users_fraction_wandb' in config_wandb:
        experiment_name = experiment_name + "_adaption_" + \
            str(config_wandb.get('app_users_fraction_wandb', -1))
            
    # if args.testing_fraction:
    #         experiment_name = experiment_name + "_testingfrac_" + str(args.testing_fraction)
            
            
    results_dir_global = (
        f'results/{experiment_name}/{configname_data}_{configname_model}/')

    # This line prints all git differences etc. No need for now.
    # util_experiments.make_git_log()

    if do_wandb:
        wandb.mark_preempting()

    util.maybe_make_dir(results_dir_global)
    if args.dump_traces:
        trace_dir_global = (
            f'../../../../scratch-shared/mturk/{args.dump_traces_folder}/trace_high_mem_{experiment_name}')
        util.maybe_make_dir(trace_dir_global)
        logger.info(f"Dump traces to results_dir_global {trace_dir_global}")
    else:
        trace_dir_global = None
    
    if args.collect_pred_data:
        trace_dir_preds = (
            f'../../../../scratch-shared/mturk/preddump/trace_{experiment_name}')
        util.maybe_make_dir(trace_dir_preds)
        logger.info(f"Dump predictions to results_dir_global {trace_dir_global}")
    else:
        trace_dir_preds = None

    logger.info(f"Logger filename {LOGGER_FILENAME}")
    logger.info(f"Saving to results_dir_global {results_dir_global}")
    logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
    logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
    logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
    logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

    static_baseline_value = -1.0

    if args.static_baseline:
        if sum([args.age_baseline, args.mean_baseline]) != 1:
            raise ValueError(
                'At least age_baseline or mean_baseline need to be selected.')
        overall_baseline = []

        for subdir, dirs, files in os.walk(args.static_baseline_path):
            for file in files:
                if file.split('.')[-1] == 'jl':
                    filename = os.path.join(subdir, file)
                    f = open(filename)
                    static_values_dict = json.load(f)

                    mean_baseline_value = static_values_dict['run_mean_baseline']
                    if mean_baseline_value > 0:
                        overall_baseline.append(mean_baseline_value)
                    else:
                        age_baseline_values = static_values_dict['running_mean_age_groups']
                        overall_baseline.append(age_baseline_values)

        static_baseline_value = np.mean(
            np.array(overall_baseline), axis=0, dtype=np.float32)
        
        logger.info(f"Static baseline value: {static_baseline_value}")

    try:
        if args.simulator == "abm":
            comparison_fn = compare_abm
        elif args.simulator == "covasim":
            comparison_fn = compare_policy_covasim
        else:
            raise ValueError(f"Unknown simulator {args.simulator}")

        # Run full comparison
        comparison_fn(
            inf_method,
            cfg=config_wandb,
            runner=runner_global,
            results_dir=results_dir_global,
            arg_rng=arg_rng,
            neural_imp_model = neural_imp_model,
            trace_dir=trace_dir_global,
            trace_dir_preds = trace_dir_preds,
            do_diagnosis=args.do_diagnosis,
            modify_contacts=args.modify_contacts,
            run_mean_baseline=args.mean_baseline,
            run_age_baseline=args.age_baseline,
            run_local_mean_baseline=args.local_mean_baseline,
            static_baseline_value=static_baseline_value,
            dl_model=dl_model,
        )

    except Exception as e:
        # This exception sends an WandB alert with the traceback and sweepid
        logger.info(f'Error repr: {repr(e)}')
        traceback_report = traceback.format_exc()
        if slurmid := os.getenv('SLURM_JOB_ID'):
            wandb.alert(
                title=f"Error {os.getenv('SWEEPID')}-{slurmid}",
                text=(
                    f"'{configname_data}', '{configname_model}', '{inf_method}'\n"
                    + traceback_report)
            )
        raise e

    runner_global.finish()
