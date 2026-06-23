#!/usr/bin/env bash
# Shared benchmark-asset location for the simulator setup scripts.
#
# The repo lives on the fast 200GB SSD (/data_fast). Large downloaded
# benchmark assets (LIBERO-Plus ~6.4GB, RoboCasa/RoboCasa365/GR-1 model
# assets, GR00T LFS) belong on the 1TB volume (/data = /storage = /data/home)
# instead. Venvs and submodule source stay on the SSD.
#
# Source this from a sim setup script (sets $RLDX_BENCH_HOME + helpers):
#   source "$SCRIPT_DIR/../_bench_env.sh"
#
# Override the location with: RLDX_BENCH_HOME=/somewhere/else bash setup_*.sh

# Default to /data/home/<username>/rldx1_bench, deriving <username> from $HOME.
: "${RLDX_BENCH_HOME:=/data/home/$(basename "$HOME")/rldx1_bench}"
export RLDX_BENCH_HOME
mkdir -p "$RLDX_BENCH_HOME"

# Move a package's asset dir onto /data and symlink it back into place, so the
# downloader writes (and the sim reads) through the symlink. Idempotent: a no-op
# once relocated. Pre-existing assets are migrated (cp -a), not re-downloaded.
#   relocate_asset_dir <pkg_assets_path> <real_subdir_name>
relocate_asset_dir() {
    local pkg_path="$1"
    local real="$RLDX_BENCH_HOME/assets/$2"
    if [ -L "$pkg_path" ]; then
        return 0
    fi
    mkdir -p "$real"
    if [ -d "$pkg_path" ]; then
        cp -a "$pkg_path/." "$real/"
        rm -rf "$pkg_path"
    fi
    ln -s "$real" "$pkg_path"
}
