#!/bin/bash
#SBATCH --job-name=robocasa365_eval
#SBATCH --partition=rtx3090,ada
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/james1990a/rldx_eval/robocasa_365/slurm/%x-%j.out

# RoboCasa365 eval on a GPU compute node. Thin SLURM wrapper around the existing
# run_scripts/eval/robocasa_365/eval_robocasa365.sh, which serves its own model
# and already supports round-robin sharding via --shard-index/--num-shards
# (auto-wired from the SLURM array env vars). Submit as an array to split the
# task set across jobs:
#   sbatch --array=0-9 run_scripts/eval/robocasa_365/eval_robocasa365_slurm.sh <MODEL_PATH> [TASK_SET] [SPLIT]
# Without --array it runs the whole task set in one job.
#
# Outputs (videos, logs, summary.csv) and the HF checkpoint go to /data (1TB),
# not the /data_fast SSD: we override the inner script's default --output-root,
# which otherwise points at the SSD.
#
# The checkpoint is ~7-8B (>=24GB VRAM), so rtx2080 is excluded from the partitions.
#
# Submit from the repo root (the --output dir must exist first):
#   mkdir -p /data/home/james1990a/rldx_eval/robocasa_365/slurm
#   sbatch [--array=0-9] run_scripts/eval/robocasa_365/eval_robocasa365_slurm.sh \
#       <MODEL_PATH> [TASK_SET] [SPLIT]

set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell
# Keep the ~16GB checkpoint download off the SSD (CLAUDE.md rule 2-2: caches -> /data).
export HF_HOME=/data/home/james1990a/.cache/huggingface
mkdir -p "$HF_HOME"
# Keep W&B quiet by default for unattended batch runs; export WANDB_ENABLED=1 to log.
export WANDB_ENABLED="${WANDB_ENABLED:-0}"

MODEL_PATH="${1:?Usage: sbatch [--array=..] eval_robocasa365_slurm.sh <MODEL_PATH> [TASK_SET] [SPLIT]}"
TASK_SET="${2:-target50}"     # atomic_seen / composite_seen / composite_unseen / target50
SPLIT="${3:-target}"          # pretrain / target

OUT_ROOT="/data/home/james1990a/rldx_eval/robocasa_365"
mkdir -p "$OUT_ROOT"

BASE_DIR="$(git rev-parse --show-toplevel)"
cd "$BASE_DIR"

find_free_port() {
  local port=$1
  while ss -lnt | awk '{print $4}' | grep -q ":$port$"; do
    port=$((port + 1)); [ "$port" -gt 65000 ] && port=20000
  done
  echo "$port"
}
PORT=$(find_free_port $((20000 + RANDOM % 40000)))

echo "[i] MODEL_PATH=$MODEL_PATH  TASK_SET=$TASK_SET  SPLIT=$SPLIT  PORT=$PORT"
echo "[i] shard (from SLURM array)=${SLURM_ARRAY_TASK_ID:-none}/${SLURM_ARRAY_TASK_COUNT:-1}  OUT=$OUT_ROOT"

# eval_robocasa365.sh serves its own model, auto-wires --shard-index/--num-shards
# from SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT, and writes under --output-root.
bash run_scripts/eval/robocasa_365/eval_robocasa365.sh \
    --model-path "$MODEL_PATH" \
    --task-set "$TASK_SET" \
    --split "$SPLIT" \
    --server-port "$PORT" \
    --output-root "$OUT_ROOT"
