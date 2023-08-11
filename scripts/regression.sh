#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --job-name=regr
#SBATCH --constraint=cpunode

source "/var/scratch/${USER}/projects/dpfn/scripts/preamble.sh"

echo `pwd`
echo "PYTHON: `which python`"
echo "WANDB: `which wandb`"
echo "SWEEP: $SWEEP, $SLURM_JOB_ID"

MESSAGE="`date "+%Y-%m-%d__%H-%M-%S"` \t ${SLURM_JOB_ID} \t ${SLURM_JOB_NAME} \t ${SWEEP}  \t "
sed -i "1i$MESSAGE" "/var/scratch/${USER}/projects/dpfn/jobs.txt"

echo 'Starting'

python3 dpfn/experiments/compare_abm.py --inference_method "fn" --experiment_setup "prequential" --config_data intermediate_graph_abm_02 --config_model model_ABM01  --name regression
