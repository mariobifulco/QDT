#!/usr/bin/env bash
#SBATCH --job-name=qdt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qdt-%A_%a.out
#SBATCH --error=logs/qdt-%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit as a job array using slurm/submit.sh}"
: "${1:?manifest path required}"
: "${2:?output directory required}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
# Install once on the login node; compute jobs never download dependencies or data.
QDT_PYTHON="${QDT_PYTHON:-$PWD/.venv/bin/python}"
srun "$QDT_PYTHON" -m qdt.campaign run --manifest "$1" \
  --index "$SLURM_ARRAY_TASK_ID" --output-dir "$2"
