#!/bin/bash
#SBATCH --job-name=robocasa365_setup
#SBATCH --partition=rtx2080,rtx3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/%x-%j.out

# Reusable SLURM job: build the RoboCasa365 eval venv on a GPU compute node.
# Builds robosuite from source, downloads the RoboCasa365 kitchen assets (to
# /data, the 1TB volume, via RLDX_BENCH_HOME), runs setup_macros.py, and ends in
# an EGL render smoke test that needs a GPU — so this must NOT run on the login
# node. Only the venv + submodule source stay on the SSD.
#
# Submit from the repo root:
#   mkdir -p slurm && sbatch run_scripts/eval/robocasa_365/setup_robocasa365_slurm.sh
#
# Override resources at submit time (CLI overrides the directives above):
#   sbatch --partition=titan,rtx2080,rtx3090 run_scripts/eval/robocasa_365/setup_robocasa365_slurm.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell

cd "$(git rev-parse --show-toplevel)"
bash run_scripts/eval/robocasa_365/setup_robocasa365.sh
