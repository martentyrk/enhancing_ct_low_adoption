#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --job-name=data_gen
#SBATCH --partition=rome
#SBATCH --mem=16000
#SBATCH --output=data_gen%A.out

source activate thesis
source scripts/preamble.sh

srun python3 dpfn/data_generation/dataset_gen_graph.py --path dpfn/data/data_all_users/frac_0.2/val --include_non_users