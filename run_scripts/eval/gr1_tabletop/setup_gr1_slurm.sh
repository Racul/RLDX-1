#!/bin/bash
#SBATCH --job-name=gr1_setup
#SBATCH --partition=rtx2080,rtx3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out

# Reusable SLURM job: build the GR-1 Tabletop eval venv on a GPU compute node.
# The setup's final smoke test renders a MuJoCo env via EGL, so it needs a GPU
# and must NOT run on the login node. Tabletop assets download to /data (the 1TB
# volume, via RLDX_BENCH_HOME); only the venv + submodule source stay on the SSD.
# Submit from the repo root:
#
#   mkdir -p slurm && sbatch run_scripts/eval/gr1_tabletop/setup_gr1_slurm.sh
#
# Override resources at submit time (CLI overrides the directives above):
#   sbatch --partition=titan,rtx2080,rtx3090 run_scripts/eval/gr1_tabletop/setup_gr1_slurm.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell

cd "$(git rev-parse --show-toplevel)"
bash run_scripts/eval/gr1_tabletop/setup_gr1.sh

# The rollout imports `rldx` (rollout_policy.py -> rldx/__init__.py eagerly loads the model
# core), which the canonical setup above does NOT install into the sim venv. Add the
# rldx-import deps, matching the proven libero_uv set (numpy/transformers/numba bumped so
# robosuite's numba stays numpy-1.26-compatible). Without this the rollout dies with
# ModuleNotFoundError (tyro/albumentations/...) and records 0 episodes.
GR1_VENV_PY="rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
uv pip install --python "$GR1_VENV_PY" \
    tyro==1.0.15 albumentations==1.4.18 diffusers==0.35.1 "accelerate>=0.34" \
    einops==0.8.1 dm_tree numpy==1.26.4 transformers==4.57.0 numba==0.65.1
