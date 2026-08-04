#!/usr/bin/env bash
# Run friday_server.py inside a throwaway ROS jazzy container.
#
#   ./scripts/friday_server_docker.sh [any friday_server flag]
#
# This host has no ROS (no /opt/ros, no rclpy), so the one ROS process in the
# stack borrows the jazzy image already on the box. The container is --rm and
# writes nothing: flask/absl-py live in the repo at .tools/ros-site, the repo
# is bind-mounted at /repo, and it runs as your uid, so no root-owned files
# appear in the tree. The long-running holiday-iekf container is NOT touched.
#
# DDS settings mirror holiday-iekf (domain 138 + cyclonedds + the thort profile);
# without them the robot's topics are invisible. Override via env if they move.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${FRIDAY_ROS_IMAGE:-cr.hday.io/rtabmap/rtabmap-jazzy:v0.1}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-138}"
RMW="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
DDS_XML="${CYCLONEDDS_XML:-/home/holiday/workspaces/holiday/tools/deploy-lab/deploy_lab/projects/base-pipeline/cyclone_dds.thort.xml}"

# Port: 5010 is the committed default (friday_server.py --flask_port and both
# SERVER_URLs in config.py). 5000 is the zot registry container on this Jetson.
#
#   --shell   drop into an interactive ROS-sourced bash instead of the server
#             (for ros2 topic list / echo / hz). Safe to run alongside the
#             server -- it gets its own container.
#
# Inside the container: do NOT `source /repo/.venv/bin/activate` (it is on the
# mount, so it is visible and tempting). It puts python3.10 first on PATH and
# rclpy -- built for the image's python3.12 -- stops importing. The venv is for
# the HOST side (teleop/env); the container side is ROS only. Likewise adb,
# the Quest, and teleop all live on the host, not in here.
ARGS=("$@")
MODE=server
if [[ "${1:-}" == "--shell" ]]; then
  MODE=shell
  ARGS=("${@:2}")
fi

if [[ ! -d "$REPO/.tools/ros-site" ]]; then
  echo "== installing server deps into .tools/ros-site (one time)"
  mkdir -p "$REPO/.tools/ros-site"
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$REPO:/repo" "$IMAGE" \
    python3 -m pip install --no-cache-dir --target /repo/.tools/ros-site flask absl-py
fi

DDS_MOUNT=()
DDS_ENV=()
if [[ -f "$DDS_XML" ]]; then
  DDS_MOUNT=(-v "$DDS_XML:/etc/ros2/cyclone_dds.xml:ro")
  DDS_ENV=(-e CYCLONEDDS_URI=file:///etc/ros2/cyclone_dds.xml)
else
  echo "warning: cyclonedds profile not found at $DDS_XML -- discovery may fail" >&2
fi

TTY=()
[[ -t 0 ]] && TTY=(-it)

if [[ "$MODE" == shell ]]; then
  NAME="friday-shell-$(id -un)-$$"
  CMD='source /opt/ros/jazzy/setup.bash
       export PYTHONPATH=/repo/.tools/ros-site:$PYTHONPATH
       cd /repo
       if [[ $# -gt 0 ]]; then exec "$@"; else exec bash; fi'
else
  NAME="friday-server-$(id -un)"
  CMD='source /opt/ros/jazzy/setup.bash
       export PYTHONPATH=/repo/.tools/ros-site:$PYTHONPATH
       exec python3 /repo/serl_robot_infra/robot_servers/friday_server.py "$@"'
fi

exec docker run --rm "${TTY[@]}" \
  --name "$NAME" \
  --network host --ipc host --pid host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION="$RMW" "${DDS_ENV[@]}" \
  "${DDS_MOUNT[@]}" -v "$REPO:/repo" \
  "$IMAGE" \
  bash -lc "$CMD" _ "${ARGS[@]}"
