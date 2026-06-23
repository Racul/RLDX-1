#!/usr/bin/env bash
set -euxo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths relative to script location
LIBERO_REPO="$SCRIPT_DIR/../../../../external_dependencies/LIBERO"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
LIBERO_UV_ENV="$SCRIPT_DIR/libero_uv"

git submodule update --init $LIBERO_REPO

# python -m pip install cmake==3.18.4
rm -rf $LIBERO_UV_ENV
mkdir -p $LIBERO_UV_ENV
uv venv $LIBERO_UV_ENV/.venv --python 3.10
source $LIBERO_UV_ENV/.venv/bin/activate
# uv pip install gymnasium==1.2.0 # -> 2.9.1 -> 
uv pip install --requirements $LIBERO_REPO/requirements.txt
uv pip install -e $LIBERO_REPO --config-settings editable_mode=compat
uv pip install --editable $PROJECT_REPO --no-deps
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.57.0 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
# Required by `import rldx` (eager model-core load): diffusers (MSAT head) and
# accelerate (backbone modeling_vtc). Installed before the numpy re-pin below so
# the numpy==1.26.4 pin still wins. Quote accelerate's spec so the shell does not
# treat ">=" as a redirection.
uv pip install diffusers==0.35.1 "accelerate>=0.34"
# robosuite 1.4.0 calls the old mj_fullM(m, dst, M) signature; mujoco>=3.2 changed it
# to mj_fullM(m, d, dst), so the unpinned transitive pull (latest 3.x) breaks
# env.reset() with "incompatible function arguments". Pin to robosuite 1.4.0's tested
# baseline (mujoco 2.3.x), overriding the version pulled by LIBERO/requirements.txt.
uv pip install mujoco==2.3.2
uv pip install numpy==1.26.4

uv pip install --editable "$PROJECT_REPO" --no-deps

# Reset the LIBERO user cache only when explicitly requested — the default
# preserves any prior LIBERO work (e.g. from a separate install).
if [ "${FORCE_CLEAN:-0}" = "1" ]; then
    rm -rf "$HOME/.libero"
fi
echo "y\n" | python -c "from rldx.eval.sim.LIBERO.libero_env import register_libero_envs"
python - <<'PY'
from rldx.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Env OK:", type(env))
PY