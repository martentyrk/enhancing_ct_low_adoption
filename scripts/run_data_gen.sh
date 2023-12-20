#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=9000
#SBATCH --output=abm_only%A.out

source activate thesis
source scripts/preamble.sh

srun python3 dpfn/dataset_manipulations.py