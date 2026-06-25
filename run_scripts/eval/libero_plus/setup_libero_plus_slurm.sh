#!/bin/bash
#SBATCH --job-name=libero_plus_setup
#SBATCH --partition=rtx2080,rtx3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

# Reusable SLURM job: build the LIBERO-Plus eval venv and download the ~6.4GB
# perturbation-asset zip. Unlike the other sim setups, LIBERO-Plus's smoke test
# only enumerates the benchmark suites (no MuJoCo render), so it needs no GPU —
# we request 0 GPUs (--gres=gpu:0) so a GPU is not held idle during the long
# download (the cluster has no dedicated CPU partition; we land on a GPU node but
# reserve no GPU). The clone, assets, and extracted files all land on /data (the
# 1TB volume, via RLDX_BENCH_HOME); only the venv stays on the SSD.
#
# This is heavy I/O + pip work and must NOT run on the login node. Submit from
# the repo root:
#   mkdir -p slurm && sbatch run_scripts/eval/libero_plus/setup_libero_plus_slurm.sh
#
# Point the download elsewhere with RLDX_BENCH_HOME / LIBERO_PLUS_DATA_DIR:
#   sbatch --export=ALL,RLDX_BENCH_HOME=/data/home/james1990a/somewhere \
#     run_scripts/eval/libero_plus/setup_libero_plus_slurm.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell

cd "$(git rev-parse --show-toplevel)"

# ImageMagick (libMagickWand) is required by `wand`, imported transitively by
# libero.libero.envs, but is not installed system-wide on this cluster. Provide it
# from a user-space micromamba env on /data (created once; idempotent), and export
# MAGICK_HOME + LD_LIBRARY_PATH so the child setup's sanity check can load it. The
# child still prepends $HOME/miniconda3/lib, which is simply absent here (harmless).
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/home/$USER/micromamba}"
IMAGEMAGICK_ENV="$MAMBA_ROOT_PREFIX/envs/imagemagick"
if ! ls "$IMAGEMAGICK_ENV"/lib/libMagickWand*.so* >/dev/null 2>&1; then
    if ! command -v micromamba >/dev/null 2>&1; then
        mkdir -p "$HOME/.local/bin"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xj -C "$HOME/.local" bin/micromamba
    fi
    micromamba create -y -p "$IMAGEMAGICK_ENV" -c conda-forge imagemagick
fi
export MAGICK_HOME="$IMAGEMAGICK_ENV"
export LD_LIBRARY_PATH="$IMAGEMAGICK_ENV/lib:${LD_LIBRARY_PATH:-}"

bash run_scripts/eval/libero_plus/setup_libero_plus.sh

# Align this venv with the proven libero_uv set. Two gaps the canonical setup leaves:
#   1. The rollout imports `rldx` (rldx/__init__.py eagerly loads the model core), but
#      diffusers + accelerate are absent and transformers must be 4.57.0 (for
#      transformers.masking_utils) -> otherwise `import rldx` ModuleNotFoundErrors.
#   2. mujoco floats to 3.x here, but robosuite 1.4.0 calls mj_fullM with the 2.3.x
#      signature -> env creation dies with "mj_fullM(): incompatible function arguments".
#      Pin mujoco==2.3.2 (same as setup_libero.sh).
LP_VENV_PY="rldx/eval/sim/LIBERO_PLUS/libero_plus_uv/.venv/bin/python"
uv pip install --python "$LP_VENV_PY" \
    diffusers==0.35.1 "accelerate>=0.34" transformers==4.57.0 mujoco==2.3.2
