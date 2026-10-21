#!/bin/bash
# One experiment on VSC (KU Leuven wICE). Resumable: resubmit if it hits the time limit.
#   Motif config:  sbatch -A <account> --job-name=X --export=ALL,CMD="notebooks/run.py configs/X.toml --split eval" vsc/job.slurm
#   Baselines:     sbatch -A <account> --job-name=B --export=ALL,CMD="notebooks/02_baselines.py --dataset M --split tuning" vsc/job.slurm
# Never two jobs on the same config/output CSV at once (rows are appended).
#SBATCH --clusters=wice
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=results/slurm_%x_%j.log

module load Python/3.12.3-GCCcore-13.3.0 Java/21.0.8   # Java: PBAD's SPMF miner
cd "$VSC_DATA/thesis"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python -u -W ignore $CMD
