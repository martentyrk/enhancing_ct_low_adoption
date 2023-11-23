#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --job-name=abm_only_model
#SBATCH --partition=rome
#SBATCH --mem=1500
#SBATCH --output=abm_only%A.out

source activate thesis
source scripts/preamble.sh


module load 2022
module load GSL/2.7-GCC-11.3.0
module load SWIG/4.0.2-GCCcore-11.3.0

srun python3 dpfn/main.py --inference_method "fn" --simulator abm --config_data intermediate_graph_abm_02 --config_model model_ABM01