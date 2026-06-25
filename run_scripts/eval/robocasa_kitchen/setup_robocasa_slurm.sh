#!/bin/bash
#SBATCH --job-name=robocasa_setup
#SBATCH --partition=rtx2080,rtx3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/%x-%j.out

# Reusable SLURM job: build the RoboCasa Kitchen eval venv on a GPU compute node.
# Builds robosuite from source, downloads the kitchen assets (to /data, the 1TB
# volume, via RLDX_BENCH_HOME), applies the seed-clamp patch, and ends in an EGL
# render smoke test that needs a GPU — so this must NOT run on the login node.
# Only the venv + submodule source stay on the SSD.
#
# Submit from the repo root:
#   mkdir -p slurm && sbatch run_scripts/eval/robocasa_kitchen/setup_robocasa_slurm.sh
#
# Override resources at submit time (CLI overrides the directives above):
#   sbatch --partition=titan,rtx2080,rtx3090 run_scripts/eval/robocasa_kitchen/setup_robocasa_slurm.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell

cd "$(git rev-parse --show-toplevel)"
# download_kitchen_assets.py confirms via input("Proceed? (y/n)") and ignores its
# -y flag; sbatch provides no stdin, so feed a 'y' (same effect as the working
# setup_RoboCasa365.sh, which pipes 'y\n' into the download script).
bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh <<< 'y'

# The rollout imports `rldx` (rollout_policy.py -> rldx/__init__.py eagerly loads the model
# core), which the canonical setup above does NOT install into the sim venv. Add the
# rldx-import deps, matching the proven libero_uv set (numpy/transformers/numba bumped so
# robosuite's numba stays numpy-1.26-compatible). Without this the rollout dies with
# ModuleNotFoundError (tyro/albumentations/...) and records 0 episodes.
RC_VENV_PY="rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
uv pip install --python "$RC_VENV_PY" \
    tyro==1.0.15 albumentations==1.4.18 diffusers==0.35.1 "accelerate>=0.34" \
    einops==0.8.1 dm_tree numpy==1.26.4 transformers==4.57.0 numba==0.65.1
