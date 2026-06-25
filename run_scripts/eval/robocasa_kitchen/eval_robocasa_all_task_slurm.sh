#!/bin/bash
#SBATCH --job-name=robocasa_eval_all
#SBATCH --partition=rtx3090,ada
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/james1990a/rldx_eval/robocasa_kitchen/slurm/%x-%j.out

# Full RoboCasa Kitchen eval on a GPU compute node: serve RLDX-1 once, then roll
# out all 24 PandaOmron kitchen tasks, recording an mp4 video per task. This is
# the single-GPU SLURM analogue of the local 4-GPU eval_robocasa.sh.
#
# Everything goes to /data (1TB), never the /data_fast SSD:
#   - videos + per-task logs : /data/home/james1990a/rldx_eval/robocasa_kitchen/<label>/<task>/
#   - HF checkpoint download  : /data/home/james1990a/.cache/huggingface  (HF_HOME)
#   - slurm job log           : /data/home/james1990a/rldx_eval/robocasa_kitchen/slurm/%x-%j.out
#
# The checkpoint is ~7-8B (>=24GB VRAM), so rtx2080 is excluded from the partitions.
# 50 episodes x 24 tasks is long; rollout_policy resumes (skips episodes already
# recorded under the same RUN_LABEL), so a job that hits the walltime can simply
# be resubmitted to continue. Tune with N_EPISODES=.. / MAX_PARALLEL=.. or
# override --time at submit.
#
# Submit from the repo root (the --output dir must exist first):
#   mkdir -p /data/home/james1990a/rldx_eval/robocasa_kitchen/slurm
#   sbatch run_scripts/eval/robocasa_kitchen/eval_robocasa_all_task_slurm.sh [RUN_LABEL] [MODEL_PATH]

set -u
export NO_ALBUMENTATIONS_UPDATE=1
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell
# Keep the ~16GB checkpoint download off the SSD (CLAUDE.md rule 2-2: caches -> /data).
export HF_HOME=/data/home/james1990a/.cache/huggingface
mkdir -p "$HF_HOME"

RUN_LABEL="${1:-robocasa_all_eval}"
MODEL_PATH="${2:-RLWRLD/RLDX-1-FT-ROBOCASA}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS=1
N_ACTION_STEPS=16
MAX_EPISODE_STEPS=720
MAX_PARALLEL="${MAX_PARALLEL:-4}"

OUT_ROOT="/data/home/james1990a/rldx_eval/robocasa_kitchen"
RUN_DIR="$OUT_ROOT/$RUN_LABEL"
mkdir -p "$RUN_DIR"

BASE_DIR="$(git rev-parse --show-toplevel)"
ROBOCASA_PY="$BASE_DIR/rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
cd "$BASE_DIR"

find_free_port() {
  local port=$1
  while ss -lnt | awk '{print $4}' | grep -q ":$port$"; do
    port=$((port + 1)); [ "$port" -gt 65000 ] && port=20000
  done
  echo "$port"
}
PORT=$(find_free_port $((20000 + RANDOM % 40000)))

echo "[i] RUN_LABEL=$RUN_LABEL  MODEL_PATH=$MODEL_PATH"
echo "[i] N_EPISODES=$N_EPISODES  N_ENVS=$N_ENVS  MAX_PARALLEL=$MAX_PARALLEL  PORT=$PORT"
echo "[i] OUT=$RUN_DIR"

# ---- model server (project .venv via uv) on the SLURM-allocated GPU ----
uv run python rldx/eval/run_rldx_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
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

TASKS=(
    "TurnSinkSpout"  "TurnOnStove"  "TurnOnSinkFaucet"  "TurnOnMicrowave"
    "TurnOffStove"  "TurnOffSinkFaucet"
    "TurnOffMicrowave"  "PnPStoveToCounter"  "PnPSinkToCounter"  "PnPMicrowaveToCounter"
    "PnPCounterToStove"  "PnPCounterToSink"
    "PnPCounterToMicrowave"  "PnPCounterToCab"  "PnPCabToCounter"  "OpenSingleDoor"
    "OpenDrawer"  "OpenDoubleDoor"
    "CoffeeSetupMug"  "CoffeeServeMug"  "CoffeePressButton"  "CloseSingleDoor"
    "CloseDrawer"  "CloseDoubleDoor"
)

TOTAL=${#TASKS[@]}
echo "[i] total tasks: $TOTAL, batching $MAX_PARALLEL at a time (server counts as one)"

RUN_PIDS=()
for TIDX in "${!TASKS[@]}"; do
    TASK="${TASKS[$TIDX]}"
    OUT="$RUN_DIR/$TASK"
    mkdir -p "$OUT"
    echo "[i] [$((TIDX + 1))/$TOTAL] $TASK (n_ep=$N_EPISODES)"
    "$ROBOCASA_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --n_episodes $N_EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --max_episode_steps $MAX_EPISODE_STEPS \
        --env_name "robocasa_panda_omron/${TASK}_PandaOmron_Env" \
        --n_action_steps $N_ACTION_STEPS \
        --n_envs $N_ENVS \
        --video_dir "$OUT" \
        >& "$OUT/eval-$TIDX.log" &
    RUN_PIDS+=($!)
    # throttle: keep at most MAX_PARALLEL background jobs in flight (server counts as one)
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 5; done
done

echo "[i] all tasks launched, waiting for completion..."
for pid in "${RUN_PIDS[@]}"; do wait "$pid"; done

echo "[i] ===== summary (success rate per task) ====="
for TIDX in "${!TASKS[@]}"; do
    TASK="${TASKS[$TIDX]}"
    SR=$(grep -iEo "success rate:?[^0-9]*[0-9.]+" "$RUN_DIR/$TASK/summary.txt" 2>/dev/null | tail -1 || echo "N/A")
    echo "[i] $TASK -> $SR"
done
echo "[i] done. videos (mp4) + logs under: $RUN_DIR"
