#!/bin/bash
#SBATCH --job-name=libero_eval_all
#SBATCH --partition=rtx3090,ada
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/data/home/james1990a/rldx_eval/libero/slurm/%x-%j.out

# Full LIBERO eval on a GPU compute node: serve RLDX-1 once, then roll out all four
# suites (spatial, object, goal, long) = 4 x 10 tasks x 2 episodes, recording an mp4
# video per task.
#
# Everything goes to /data (1TB), never the /data_fast SSD:
#   - videos + per-task logs : /data/home/james1990a/rldx_eval/libero/<label>/<suite>/<task>/
#   - HF checkpoint download  : /data/home/james1990a/.cache/huggingface  (HF_HOME)
#   - slurm job log           : /data/home/james1990a/rldx_eval/libero/slurm/%x-%j.out
#
# The checkpoint is ~7-8B (>=24GB VRAM), so rtx2080 is excluded from the partitions.
#
# Submit from the repo root (the --output dir must exist first):
#   mkdir -p /data/home/james1990a/rldx_eval/libero/slurm
#   sbatch run_scripts/eval/libero/eval_libero_all_task_slurm.sh [RUN_LABEL] [MODEL_PATH]

set -u
export NO_ALBUMENTATIONS_UPDATE=1
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under sbatch's non-login shell
# Keep the ~16GB checkpoint download off the SSD (CLAUDE.md rule 2-2: caches -> /data).
export HF_HOME=/data/home/james1990a/.cache/huggingface
mkdir -p "$HF_HOME"

RUN_LABEL="${1:-libero_all_eval}"
MODEL_PATH="${2:-RLWRLD/RLDX-1-FT-LIBERO}"
N_EPISODES=2
N_ENVS=2
N_ACTION_STEPS=8
MAX_EPISODE_STEPS=720
MAX_PARALLEL=4

OUT_ROOT="/data/home/james1990a/rldx_eval/libero"
RUN_DIR="$OUT_ROOT/$RUN_LABEL"
mkdir -p "$RUN_DIR"

BASE_DIR="$(git rev-parse --show-toplevel)"
LIBERO_PY="$BASE_DIR/rldx/eval/sim/LIBERO/libero_uv/.venv/bin/python"
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

LIBERO_SPATIAL_TASKS=(
    "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)
LIBERO_OBJECT_TASKS=(
    "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_butter_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_milk_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket"
)
LIBERO_GOAL_TASKS=(
    "libero_sim/open_the_middle_drawer_of_the_cabinet"
    "libero_sim/put_the_bowl_on_the_stove"
    "libero_sim/put_the_wine_bottle_on_top_of_the_cabinet"
    "libero_sim/open_the_top_drawer_and_put_the_bowl_inside"
    "libero_sim/put_the_bowl_on_top_of_the_cabinet"
    "libero_sim/push_the_plate_to_the_front_of_the_stove"
    "libero_sim/put_the_cream_cheese_in_the_bowl"
    "libero_sim/turn_on_the_stove"
    "libero_sim/put_the_bowl_on_the_plate"
    "libero_sim/put_the_wine_bottle_on_the_rack"
)
LIBERO_10_TASKS=(
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
    "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
    "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
    "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
    "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
    "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)

ALL_TASKS=() ALL_SUITES=() ALL_TASK_INDICES=()
for i in "${!LIBERO_SPATIAL_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_SPATIAL_TASKS[$i]}"); ALL_SUITES+=("libero_spatial"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_OBJECT_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_OBJECT_TASKS[$i]}"); ALL_SUITES+=("libero_object"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_GOAL_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_GOAL_TASKS[$i]}"); ALL_SUITES+=("libero_goal"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_10_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_10_TASKS[$i]}"); ALL_SUITES+=("libero_10"); ALL_TASK_INDICES+=($i)
done

TOTAL=${#ALL_TASKS[@]}
echo "[i] total tasks: $TOTAL (4 suites x 10), batching $MAX_PARALLEL at a time"

RUN_PIDS=()
for idx in "${!ALL_TASKS[@]}"; do
    TASK="${ALL_TASKS[$idx]}"
    SUITE="${ALL_SUITES[$idx]}"
    TIDX="${ALL_TASK_INDICES[$idx]}"
    CLEAN="${TASK#libero_sim/}"
    OUT="$RUN_DIR/$SUITE/$CLEAN"
    mkdir -p "$OUT"
    echo "[i] [$((idx + 1))/$TOTAL] $SUITE :: $CLEAN (n_ep=$N_EPISODES)"
    "$LIBERO_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --n_episodes $N_EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --max_episode_steps $MAX_EPISODE_STEPS \
        --env_name "$TASK" \
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

echo "[i] ===== summary (success_rate per task) ====="
for idx in "${!ALL_TASKS[@]}"; do
    SUITE="${ALL_SUITES[$idx]}"
    CLEAN="${ALL_TASKS[$idx]#libero_sim/}"
    TIDX="${ALL_TASK_INDICES[$idx]}"
    LOG="$RUN_DIR/$SUITE/$CLEAN/eval-$TIDX.log"
    SR=$(grep -oE "success rate[^0-9]*[0-9.]+" "$LOG" 2>/dev/null | tail -1 || echo "N/A")
    echo "[i] $SUITE :: $CLEAN -> $SR"
done
echo "[i] done. videos (mp4) + logs under: $RUN_DIR"
