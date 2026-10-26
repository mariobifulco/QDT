#!/usr/bin/env bash
#SBATCH --job-name=qdt-collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/collect-%j.out

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
QDT_PYTHON="${QDT_PYTHON:-$PWD/.venv/bin/python}"
srun "$QDT_PYTHON" -m qdt.campaign collect --manifest "${1:?manifest required}" \
  --output-dir "${2:?output directory required}"
