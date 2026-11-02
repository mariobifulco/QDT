#!/usr/bin/env bash
#SBATCH --job-name=qdt-workflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/qdt-workflow-%j.out

# Submit from the repository root after mkdir -p logs.
# Override partition/account/CPUs/GPUs at submission; set QDT_PYTHON for a GPU environment.
set -euo pipefail
: "${1:?Pass a qdt command, for example train or campaign}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
QDT_PYTHON="${QDT_PYTHON:-$PWD/.venv/bin/python}"
srun "$QDT_PYTHON" -m qdt "$@"
