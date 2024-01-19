#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=9000
#SBATCH --output=data_gen%A.out

source activate thesis
source scripts/preamble.sh

srun python3 dpfn/dataset_generator.py --path dpfn/data/test_app_users