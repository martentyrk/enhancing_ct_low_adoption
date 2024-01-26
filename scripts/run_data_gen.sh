#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=9000
#SBATCH --output=data_gen%A.out

source activate thesis
source scripts/preamble.sh

srun python3 dpfn/data_generation/dataset_gen_graph.py --path dpfn/data/data_all_users/frac_0.6/val --include_non_users