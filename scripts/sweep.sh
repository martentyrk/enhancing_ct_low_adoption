#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=CPUS
#SBATCH --ntasks=NTASKS
#SBATCH --time=96:00:00
#SBATCH --job-name=sweep
#SBATCH --constraint=cpunode
#\ #SBATCH --partition=fatq

source /var/scratch/rromijnd/projects/dpfn/scripts/preamble.sh

echo `pwd`
echo "PYTHON: `which python`"
echo "WANDB: `which wandb`"
echo "SWEEP: $SWEEP, $SLURM_JOB_ID"

MESSAGE="`date "+%Y-%m-%d__%H-%M-%S"` \t $SLURM_JOB_ID \t $SLURM_JOB_NAME \t "
sed -i "1i$MESSAGE" '/var/scratch/rromijnd/projects/dpfn/jobs.txt'

echo 'Starting'
export PYTHON='/var/scratch/rromijnd/conda/envs/ml38/bin/python3'

export SWEEPID=$SWEEP

for i in {1..NTASKS}
do
   wandb agent "robromijnders/dpfn-dpfn_experiments/$SWEEP" &
done

wait
