#!/bin/bash
#SBATCH --job-name=libero_plus_eval
#SBATCH --partition=rtx3090,ada
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/james1990a/rldx_eval/libero_plus/slurm/%x-%j.out

# Full LIBERO-Plus eval on a GPU compute node: serve RLDX-1 once, then roll out
# the perturbation tasks (1 episode each), recording an mp4 video per task.
#
# LIBERO-Plus enumerates thousands of perturbed tasks — far too many for one GPU.
# Submit as a SLURM array to shard the task list round-robin across jobs (each
# array task serves its own model and runs the tasks where
# task_idx % NUM_SHARDS == SHARD_INDEX, auto-wired from the array env vars):
#   sbatch --array=0-49 run_scripts/eval/libero_plus/eval_libero_plus_all_task_slurm.sh
# Without --array it runs every task in a single job (NUM_SHARDS=1).
#
# Everything goes to /data (1TB), never the /data_fast SSD:
#   - videos + per-task logs : /data/home/james1990a/rldx_eval/libero_plus/<label>/<suite>/<task>/
#   - HF checkpoint download  : /data/home/james1990a/.cache/huggingface  (HF_HOME)
#   - slurm job log           : /data/home/james1990a/rldx_eval/libero_plus/slurm/%x-%j.out
#
# The checkpoint is ~7-8B (>=24GB VRAM), so rtx2080 is excluded from the partitions.
#
# Submit from the repo root (the --output dir must exist first):
#   mkdir -p /data/home/james1990a/rldx_eval/libero_plus/slurm
#   sbatch [--array=0-49] run_scripts/eval/libero_plus/eval_libero_plus_all_task_slurm.sh \
#       [RUN_LABEL] [MODEL_PATH] [SUITE_FILTER]

set -u
export NO_ALBUMENTATIONS_UPDATE=1
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell
# Keep the ~16GB checkpoint download off the SSD (CLAUDE.md rule 2-2: caches -> /data).
export HF_HOME=/data/home/james1990a/.cache/huggingface
mkdir -p "$HF_HOME"

RUN_LABEL="${1:-libero_plus_all_eval}"
MODEL_PATH="${2:-RLWRLD/RLDX-1-FT-LIBERO}"
SUITE_FILTER="${3:-}"   # optional: libero_10 / libero_spatial / libero_object / libero_goal
N_EPISODES="${N_EPISODES:-1}"   # LIBERO-Plus paper: 1 trial per perturbed task
N_ENVS=1
N_ACTION_STEPS=8
MAX_EPISODE_STEPS=720
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# Round-robin shard config, auto-wired from the SLURM array (defaults: 1 shard).
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_TAG="shard${SHARD_INDEX}of${NUM_SHARDS}"

OUT_ROOT="/data/home/james1990a/rldx_eval/libero_plus"
RUN_DIR="$OUT_ROOT/$RUN_LABEL"
mkdir -p "$RUN_DIR"

BASE_DIR="$(git rev-parse --show-toplevel)"
cd "$BASE_DIR"
source "$BASE_DIR/rldx/eval/sim/_bench_env.sh"
LIBERO_PY="$BASE_DIR/rldx/eval/sim/LIBERO_PLUS/libero_plus_uv/.venv/bin/python"
LIBERO_PLUS_REPO="${LIBERO_PLUS_DATA_DIR:-$RLDX_BENCH_HOME/LIBERO-plus}"
export LIBERO_CONFIG_PATH="$LIBERO_PLUS_REPO/.libero_config"
# wand (imported transitively by libero.libero.envs during task enumeration AND when the
# rollout builds the env) needs the ImageMagick (MagickWand) shared lib, provided
# user-space by the micromamba env that setup_libero_plus_slurm.sh creates (no apt here).
# Mirror that setup's MAGICK_HOME + LD_LIBRARY_PATH so wand can find libMagickWand and
# resolve its libMagickCore dep in-process. (The previous $HOME/miniconda3/lib path does
# not exist on this cluster, which is what made enumeration crash -> 0 tasks.)
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/home/$USER/micromamba}"
IMAGEMAGICK_ENV="$MAMBA_ROOT_PREFIX/envs/imagemagick"
export MAGICK_HOME="$IMAGEMAGICK_ENV"
export LD_LIBRARY_PATH="$IMAGEMAGICK_ENV/lib:${LD_LIBRARY_PATH:-}"

find_free_port() {
  local port=$1
  while ss -lnt | awk '{print $4}' | grep -q ":$port$"; do
    port=$((port + 1)); [ "$port" -gt 65000 ] && port=20000
  done
  echo "$port"
}
PORT=$(find_free_port $((20000 + RANDOM % 40000)))

echo "[i] RUN_LABEL=$RUN_LABEL  MODEL_PATH=$MODEL_PATH  SUITE_FILTER=${SUITE_FILTER:-all}"
echo "[i] N_EPISODES=$N_EPISODES  MAX_PARALLEL=$MAX_PARALLEL  shard=$SHARD_INDEX/$NUM_SHARDS  PORT=$PORT"
echo "[i] OUT=$RUN_DIR"

# ---- model server (project .venv via uv) on the SLURM-allocated GPU ----
uv run python rldx/eval/run_rldx_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --no-strict \
    --host 127.0.0.1 \
    --port "$PORT" &
SERVE_PID=$!
trap 'echo "[i] killing server PID=$SERVE_PID"; kill $SERVE_PID 2>/dev/null' EXIT

echo "[i] waiting for server readiness (model load + possible ~16GB download)..."
for i in $(seq 1 1800); do
  if ss -lnt | awk '{print $4}' | grep -q ":$PORT$"; then
    echo "[i] server listening on :$PORT after ${i}s"; break
  fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "[!] server died before binding :$PORT"; exit 1
  fi
  sleep 1
done
sleep 5  # settle past port-open

# ---- enumerate (suite, task) pairs from the LIBERO-Plus benchmark ----
TASK_LIST=$("$LIBERO_PY" - "$SUITE_FILTER" <<'PYEOF'
import contextlib, io, sys
from libero.libero import benchmark

suite_filter = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
suites = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
if suite_filter:
    suites = [s for s in suites if s == suite_filter]

with contextlib.redirect_stdout(io.StringIO()):
    bd = benchmark.get_benchmark_dict()
    suite_objs = {s: bd[s]() for s in suites if s in bd}

for name, suite in suite_objs.items():
    for task_id in range(suite.get_num_tasks()):
        print(f"{name}\t{suite.get_task(task_id).name}")
PYEOF
)

IFS=$'\n' read -r -d '' -a TASK_LINES <<< "$TASK_LIST" || true
TOTAL=${#TASK_LINES[@]}
echo "[i] enumerated $TOTAL tasks; shard $SHARD_INDEX/$NUM_SHARDS runs task_idx %% NUM_SHARDS == SHARD_INDEX"
echo "[i] batching $MAX_PARALLEL at a time (server counts as one)"

RUN_PIDS=()
for idx in "${!TASK_LINES[@]}"; do
    (( idx % NUM_SHARDS == SHARD_INDEX )) || continue
    SUITE="${TASK_LINES[$idx]%%$'\t'*}"
    TASK_NAME="${TASK_LINES[$idx]#*$'\t'}"
    OUT="$RUN_DIR/$SUITE/$TASK_NAME"
    mkdir -p "$OUT"
    echo "[i] [$((idx + 1))/$TOTAL] $SUITE / $TASK_NAME (n_ep=$N_EPISODES)"
    "$LIBERO_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --n_episodes $N_EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --max_episode_steps $MAX_EPISODE_STEPS \
        --env_name "libero_plus_sim/$TASK_NAME" \
        --n_action_steps $N_ACTION_STEPS \
        --n_envs $N_ENVS \
        --video_dir "$OUT" \
        >& "$OUT/eval-$SHARD_TAG.log" &
    RUN_PIDS+=($!)
    # throttle: keep at most MAX_PARALLEL background jobs in flight (server counts as one)
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 5; done
done

echo "[i] shard tasks launched, waiting for completion..."
for pid in "${RUN_PIDS[@]}"; do wait "$pid"; done

echo "[i] ===== summary (success rate per task, shard $SHARD_INDEX/$NUM_SHARDS) ====="
for idx in "${!TASK_LINES[@]}"; do
    (( idx % NUM_SHARDS == SHARD_INDEX )) || continue
    SUITE="${TASK_LINES[$idx]%%$'\t'*}"
    TASK_NAME="${TASK_LINES[$idx]#*$'\t'}"
    SR=$(grep -iEo "success rate:?[^0-9]*[0-9.]+" "$RUN_DIR/$SUITE/$TASK_NAME/summary.txt" 2>/dev/null | tail -1 || echo "N/A")
    echo "[i] $SUITE / $TASK_NAME -> $SR"
done
echo "[i] done. videos (mp4) + logs under: $RUN_DIR"
