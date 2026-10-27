#!/usr/bin/env bash
#SBATCH --job-name=qdt
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qdt-%A_%a.out

# From the repo root: mkdir -p logs
# sbatch slurm/run.sbatch configs/default.json results/pilot
# Override --array, --partition and --account on the sbatch command as needed.
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
QDT_PYTHON="${QDT_PYTHON:-$PWD/.venv/bin/python}"
srun "$QDT_PYTHON" -m qdt --config "${1:?config required}" \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --output "${2:?output directory required}/${SLURM_ARRAY_TASK_ID}.json"
