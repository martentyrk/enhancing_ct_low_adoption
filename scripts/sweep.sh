#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=CPUS
#SBATCH --ntasks=NTASKS
#SBATCH --time=40:00:00
#SBATCH --job-name=marten_sweep
#SBATCH --partition=rome
#SBATCH --mem=90000
#SBATCH --output=sweep_%A.out

source activate thesis
source scripts/preamble.sh

export SWEEPID=WANDBSWEEP

module load 2022
module load GSL/2.7-GCC-11.3.0
module load SWIG/4.0.2-GCCcore-11.3.0

echo `pwd`
echo "PYTHON: `which python`"
echo "WANDB: `which wandb`"
echo "SWEEP: ${SWEEPID}, $SLURM_JOB_ID"

echo 'Starting'

for i in {1..NTASKS}
do
   wandb agent "martentyrk/dpfn-dpfn/${SWEEPID}" &
done

wait
