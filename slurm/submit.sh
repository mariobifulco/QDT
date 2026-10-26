#!/usr/bin/env bash
# Run from the repository root. Extra sbatch options apply to both submissions.
set -euo pipefail
MANIFEST="${1:?Usage: bash slurm/submit.sh MANIFEST OUTPUT [sbatch options]}"
OUTPUT="${2:?output directory required}"
shift 2
QDT_PYTHON="${QDT_PYTHON:-$PWD/.venv/bin/python}"
COUNT=$("$QDT_PYTHON" -m qdt.campaign count --manifest "$MANIFEST")
mkdir -p logs "$OUTPUT"
# Use absolute shared-filesystem paths; relative dataset paths resolve from the repo.
ROOT="$PWD"
MANIFEST=$("$QDT_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$MANIFEST")
OUTPUT=$("$QDT_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OUTPUT")
JOB=$(sbatch --parsable --chdir="$ROOT" --array="0-$((COUNT-1))%${QDT_CONCURRENCY:-16}" \
  "$@" slurm/run.sbatch "$MANIFEST" "$OUTPUT")
JOB="${JOB%%;*}"
echo "Submitted array $JOB ($COUNT runs)"
sbatch "$@" --chdir="$ROOT" --dependency="afterok:$JOB" slurm/collect.sbatch "$MANIFEST" "$OUTPUT"
