#!/bin/bash
# Submit an eval SLURM script after ensuring its `#SBATCH --output` directory exists.
#
# Why this exists: sbatch evaluates `--output` and opens the job's stdout file BEFORE
# the script body runs, so an eval script cannot mkdir its own output dir. If that dir
# is missing, slurmstepd fails to open stdout and the job is CANCELLED at 0s with NO
# log written (State=FAILED, batch=CANCELLED, ExitCode 0:53). This wrapper creates the
# dir first, then submits, so plain "forgot to mkdir" never silently kills a job again.
#
# Usage (from the repo root):
#   bash run_scripts/eval/submit.sh <eval_*_slurm.sh> [extra sbatch/script args...]
# Examples:
#   bash run_scripts/eval/submit.sh run_scripts/eval/gr1_tabletop/eval_gr1_all_task_slurm.sh
#   bash run_scripts/eval/submit.sh run_scripts/eval/robocasa_365/eval_robocasa365_slurm.sh <MODEL_PATH>
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: $0 <eval_*_slurm.sh> [args...]" >&2; exit 1; }
script=$1; shift

out_tpl=$(awk -F= '/^#SBATCH[[:space:]]+--output=/{print $2; exit}' "$script")
[ -n "$out_tpl" ] || { echo "[submit] no '#SBATCH --output=' line in $script" >&2; exit 1; }

out_dir=$(dirname "$out_tpl")   # strips the %x-%j.out filename; %-tokens only live in the filename
mkdir -p "$out_dir"
echo "[submit] output dir ready: $out_dir"
exec sbatch "$script" "$@"
