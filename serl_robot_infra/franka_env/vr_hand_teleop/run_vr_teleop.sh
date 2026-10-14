#!/usr/bin/env bash
# Quest hand tracking -> XHand finger targets + wrist poses, via remy2 vr_teleop
# run in remy2's own venv (py3.12 + torch-cpu + pinocchio; not the hilserl conda).
#
#   adb reverse tcp:8000 tcp:8000
#   serl_robot_infra/franka_env/vr_hand_teleop/run_vr_teleop.sh                # ZMQ :5555
#   serl_robot_infra/franka_env/vr_hand_teleop/run_vr_teleop.sh --publisher ros2  # holiday/manipulation (needs friday_msgs)
#
# REMY2=/path/to/remy2 overrides the checkout (default ~/workspace/remy2); run
# `uv sync` there once. remy2 is not modified: see run_vr_teleop.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMY2="${REMY2:-$HOME/workspace/remy2}"
export REMY2
exec uv run --project "$REMY2/vr_teleop" --no-sync python "$HERE/run_vr_teleop.py" "$@"
