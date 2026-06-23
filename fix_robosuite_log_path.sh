#!/usr/bin/env bash
# Redirect robosuite's hardcoded file-log path to a per-user location.
#
# robosuite/utils/log_utils.py opens logging.FileHandler("/tmp/robosuite.log")
# at import time (file logging is on by default, FILE_LOGGING_LEVEL="DEBUG").
# On this shared cluster that path is owned by whichever user created it first,
# so `import robosuite` dies with:
#     PermissionError: [Errno 13] Permission denied: '/tmp/robosuite.log'
#
# This patches the path to /tmp/$USER/robosuite.log. The directory is created at
# runtime (os.makedirs(..., exist_ok=True)) so it also works on SLURM compute
# nodes, where /tmp is node-local and ephemeral. Idempotent: safe to re-run.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PY="$SCRIPT_DIR/rldx/eval/sim/LIBERO_PLUS/libero_plus_uv/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] LIBERO-Plus venv python not found at $VENV_PY"
    echo "        Run setup_libero_plus.sh first."
    exit 1
fi

# Resolve the installed robosuite log_utils.py from inside the venv WITHOUT
# importing robosuite (importing it is exactly what triggers the error we fix).
LOG_UTILS="$("$VENV_PY" -c 'import importlib.util, os; s = importlib.util.find_spec("robosuite"); print(os.path.join(os.path.dirname(s.origin), "utils", "log_utils.py"))')"

"$VENV_PY" - "$LOG_UTILS" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

old = 'fh = logging.FileHandler("/tmp/robosuite.log")'
new = (
    'import os, getpass\n'
    '            _log_dir = os.path.join("/tmp", getpass.getuser())\n'
    '            os.makedirs(_log_dir, exist_ok=True)\n'
    '            fh = logging.FileHandler(os.path.join(_log_dir, "robosuite.log"))'
)
marker = 'fh = logging.FileHandler(os.path.join(_log_dir, "robosuite.log"))'

if marker in src:
    print(f"  Already patched: {path}")
elif old in src:
    open(path, "w").write(src.replace(old, new))
    print(f"  Patched robosuite log path -> /tmp/$USER/robosuite.log in {path}")
else:
    print(f"[ERROR] Expected line not found in {path}; robosuite may have changed.")
    sys.exit(1)
PYEOF

# Verify: importing robosuite is what triggered the original PermissionError.
echo "  Verifying 'import robosuite'..."
"$VENV_PY" -c "import robosuite; print('  OK: robosuite imported, log dir = /tmp/' + __import__('getpass').getuser())"

echo "Done."
