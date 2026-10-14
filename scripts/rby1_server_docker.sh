#!/usr/bin/env bash
# Run robot_servers/rby1_server.py against the holiday rby1 stack (fabrics
# controller + rby1-driver), in SIL or on the real robot.
#
#   scripts/rby1_server_docker.sh                  # server on :5010 (host network)
#   scripts/rby1_server_docker.sh --shell          # bash with ROS + friday_msgs sourced
#   scripts/rby1_server_docker.sh --shell ros2 topic hz /holiday/joint_states
#
# Stack first (holiday checkout, branch feat/rby1-driver):
#   src/drivers/rby1-driver/sil.sh up && sil.sh enable       # SIL
#   just rt run rby1_xhand; torque on; control on             # robot
# Same ROS domain (138) and host network as that stack. The image is the
# driver image + flask/scipy/absl (scripts/Dockerfile.rby1_server).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${RBY1_DRIVER_IMAGE:-rby1-driver:jazzy}"
IMAGE="${RBY1_SERVER_IMAGE:-rby1-server:jazzy}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-138}"
NAME="rby1-server-$USER"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "== building $IMAGE from $BASE_IMAGE (one time)"
  docker build --network host --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$IMAGE" -f "$REPO/scripts/Dockerfile.rby1_server" "$REPO/scripts"
fi

TTY=(); [[ -t 0 ]] && TTY=(-it)
COMMON=(--rm "${TTY[@]}" --network host --ipc host
        -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e CYCLONEDDS_URI=
        -v "$REPO:/repo" -w /repo)
SRC='source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash'

if [[ "${1:-}" == "--shell" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    exec docker run "${COMMON[@]}" "$IMAGE" bash -c "$SRC && exec bash"
  fi
  exec docker run "${COMMON[@]}" "$IMAGE" bash -c "$SRC && exec $(printf '%q ' "$@")"
fi
exec docker run --name "$NAME" "${COMMON[@]}" "$IMAGE" \
  bash -c "$SRC && exec python3 /repo/serl_robot_infra/robot_servers/rby1_server.py $(printf '%q ' "$@")"
