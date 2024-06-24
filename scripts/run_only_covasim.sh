#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=46:00:00
#SBATCH --job-name=cov_only_model
#SBATCH --partition=rome
#SBATCH --mem=128000
#SBATCH --output=covasim_only%A.out

source activate thesis
source scripts/preamble.sh


module load 2022
module load GSL/2.7-GCC-11.3.0
module load SWIG/4.0.2-GCCcore-11.3.0

srun python3 dpfn/main.py \
 --simulator covasim \
 --config_data intermediate_graph_cv_01 \
 --config_model model_CV01 \
 --app_users_fraction 0.4 \
 --name gcn_cov_raw_250k_no_onehot \
 --num_users 500000 \
 --seed_value 40 \
 --inference_method "fn" \
 --model_name=500k/gcn_250k_cov_raw_no_onehot_final.pth \
 --model gcn \
#  --one_hot_encoding \