#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=03:30:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=9000
#SBATCH --output=abm_only%A.out

source activate thesis
source scripts/preamble.sh


module load 2022
module load GSL/2.7-GCC-11.3.0
module load SWIG/4.0.2-GCCcore-11.3.0

# srun python3 dpfn/main.py --model set --name=10k_users_SET_06_test --model_name set_all_users_0.6_1layers.pth --seed_value 45 --app_users_fraction 0.6  --n_layers 1  --config_data=intermediate_graph_abm_02 --simulator=abm --inference_method=fn --num_users 10000

srun python3 dpfn/main.py --app_users_fraction 0.6 --name=seed30_linreg_test --feature_imp_model 'dpfn/config/feature_imp_configs/lin_reg_08data.joblib' --mean_baseline --num_users 100000 --seed_value 30 --inference_method "fn" --simulator abm --config_data intermediate_graph_abm_02 --config_model model_ABM01