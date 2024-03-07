#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=9000
#SBATCH --output=covasim_only%A.out

source activate thesis
source scripts/preamble.sh


module load 2022
module load GSL/2.7-GCC-11.3.0
module load SWIG/4.0.2-GCCcore-11.3.0

srun python3 dpfn/main.py --app_users_fraction 0.6 --name=cov_test --num_users 10000 --seed_value 99 --inference_method "fn" --simulator covasim --config_data intermediate_graph_cv_01 --config_model model_CV01