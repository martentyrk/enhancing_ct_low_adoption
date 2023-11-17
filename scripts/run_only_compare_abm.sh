#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=00:25:00
#SBATCH --job-name=marten_sweep
#SBATCH --partition=rome
#SBATCH --mem=2000
#SBATCH --output=abm_only_%A.out

source activate thesis
source scripts/preamble.sh


module load 2022
module load GSL/2.7-GCC-11.3.0
module load SWIG/4.0.2-GCCcore-11.3.0

srun python3 dpfn/experiments/compare_abm.py --inference_method "fncpp" --simulator abm --config_data intermediate_graph_abm_02 --config_model model_ABM01