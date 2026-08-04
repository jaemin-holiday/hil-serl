#!/usr/bin/env bash
# Self-contained install for the Jetson / any host, entirely inside this repo.
#
#   ./scripts/setup_venv.sh
#
# Creates:
#   .tools/bin      uv (repo-local, not on your PATH, no shell profile touched)
#   .tools/python   uv-managed CPython 3.10 (system python stays untouched)
#   .tools/cache    uv download cache
#   .venv           the teleop / env / RL-client virtualenv
#
# Nothing is written outside this directory: no apt, no sudo, no conda, no
# ~/.local, no ~/.cache. Delete the four dirs above to fully uninstall.
#
# ROS is NOT installed here -- friday_server.py needs rclpy and runs in a
# jazzy container instead: scripts/friday_server_docker.sh.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export UV_UNMANAGED_INSTALL="$REPO/.tools/bin"      # install dir + no PATH edits
export UV_PYTHON_INSTALL_DIR="$REPO/.tools/python"
export UV_CACHE_DIR="$REPO/.tools/cache"
export VIRTUAL_ENV="$REPO/.venv"
UV="$REPO/.tools/bin/uv"

if [[ ! -x "$UV" ]]; then
  echo "== installing uv into $REPO/.tools/bin"
  mkdir -p "$REPO/.tools/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "== creating .venv (CPython 3.10, matches the documented conda env)"
"$UV" venv --python 3.10 "$REPO/.venv"

# Deps are listed explicitly rather than taken from the packages' install_requires:
# upstream pulls pymodbus/hidapi/rospkg/tensorflow and the bogus PyPI `typing`
# and `zmq` shim packages, none of which the friday stack needs on aarch64.
echo "== installing runtime deps"
"$UV" pip install \
  "numpy<2" scipy requests flask absl-py defusedxml pyyaml pyquaternion \
  opencv-python "gymnasium==0.29.1" "gym>=0.26" pynput tqdm natsort matplotlib imageio \
  easyhid pyrealsense2 \
  jax flax optax distrax chex einops ml_collections lz4 pyzmq

echo "== installing this repo (editable, --no-deps on purpose)"
"$UV" pip install --no-deps -e serl_launcher -e serl_robot_infra

echo "== smoke test"
"$REPO/.venv/bin/python" - <<'EOF'
import pathlib, sys
sys.path.insert(0, str(pathlib.Path("examples").resolve()))
import jax
from experiments.friday_reach.config import TrainConfig  # pulls the whole chain
print("OK  jax", jax.__version__, "backend", jax.default_backend())
EOF

cat <<EOF

Done. Use it with:
    source $REPO/.venv/bin/activate

Headless (no X): nothing to set. Do NOT export PYNPUT_BACKEND=dummy -- the
ESC-listener guard in friday_arm_env.py relies on the ImportError, and the
dummy backend moves the failure into a thread where the guard cannot catch it
(one spurious traceback per arm).

Learner-only extras (GPU workstation, NOT this Jetson):
    $UV pip install tensorflow tensorflow_probability wandb moviepy \\
      "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d"
EOF
