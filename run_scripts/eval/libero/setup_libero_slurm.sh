#!/bin/bash
#SBATCH --job-name=libero_setup
#SBATCH --partition=rtx2080,rtx3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x-%j.out

# Reusable SLURM job: build the LIBERO eval venv on a GPU compute node.
# The setup's final env smoke test (MuJoCo EGL offscreen render) needs a GPU, so
# this must NOT run on the login node. Submit from the repo root:
#
#   mkdir -p slurm && sbatch run_scripts/eval/libero/setup_libero_slurm.sh
#
# Override resources at submit time (CLI overrides the directives above):
#   sbatch --partition=titan,rtx2080,rtx3090 run_scripts/eval/libero/setup_libero_slurm.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell

cd "$(git rev-parse --show-toplevel)"
bash run_scripts/eval/libero/setup_libero.sh
