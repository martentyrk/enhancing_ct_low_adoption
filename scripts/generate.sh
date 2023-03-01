#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --job-name=GEN-IG03

source /var/scratch/${USER}/projects/dpfn/scripts/preamble.sh

PYTHON="/var/scratch/${USER}/conda/envs/ml38/bin/python3"

MESSAGE="`date "+%Y-%m-%d__%H-%M-%S"` \t ${SLURM_JOB_ID} \t ${SLURM_JOB_NAME} \t ${SWEEP}  \t "
sed -i "1i$MESSAGE" "/var/scratch/${USER}/projects/dpfn/jobs.txt"

$PYTHON dpfn/data/generate_graph.py --config intermediate_graph_03 --sample_contact
