#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=9000
#SBATCH --output=data_gen%A.out

source activate thesis
source scripts/preamble.sh

srun python3 dpfn/dataset_generator.py --path dpfn/data/train_app_users/partial --out_name train_dataset.pt