#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=9800
#SBATCH --output=abm_only%A.out

source activate thesis
source scripts/preamble.sh


module load 2022
module load GSL/2.7-GCC-11.3.0
module load SWIG/4.0.2-GCCcore-11.3.0

# srun python3 dpfn/main.py --model set --name=10k_users_SET_06_test --model_name set_all_users_0.6_1layers.pth --seed_value 45 --app_users_fraction 0.6  --n_layers 1  --config_data=intermediate_graph_abm_02 --simulator=abm --inference_method=fn --num_users 10000

srun python3 dpfn/main.py \
    --app_users_fraction 0.6 \
    --name=test_run_for_cpu \
    --num_users 10000 \
    --seed_value 45 \
    --inference_method "fn" \
    --simulator abm \
    --config_data intermediate_graph_abm_02 \
    --config_model model_ABM01 \
    # --feature_imp_model=abm_linreg_1data_local_avg.joblib
    # --model_name=500k/gcn_250k_abm_meanbase_final.pth \
    # --model gcn \
    # --mean_baseline \