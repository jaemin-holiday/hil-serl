# RB-Y1 + XHand Runbook

Operational runbook for the Rainbow RB-Y1 (v1.2) + 2× RobotEra XHand1 build,
robot type `rby1_xhand`. Covers the two things that run today: **VR teleop in
sim** and **HIL-SERL against the holiday controller** (vendor simulator now,
the robot later). Companion: `examples/experiments/rby1_reach/README.md`,
`docs/friday_runbook.md` (the previous robot; the hil-serl side is the same).

Status 2026-10-14: robot not assembled. Everything below was verified in
simulation on this PC (RTX 5080, Ubuntu 22.04 / glibc 2.35, ROS humble on the
host only — every ROS jazzy piece runs in a container).

## 1. Repos and branches

| Repo | Path | Branch | What it carries |
|---|---|---|---|
| holiday (monorepo) | `~/workspace/holiday-rby1` (worktree) | `feat/rby1-driver` (+ local commits) | RB-Y1 driver, holiday controller (fabrics) for `rby1_xhand`, SIL (`sil.sh`), specs, URDF |
| remy2 | `~/workspace/remy2` | `feat/rby1-xhand` | vr_teleop `--robot rby1_xhand` (XHand IK profile, wrist/routing configs), rby1-sdk follower |
| holiday-lab | `~/workspace/holiday-lab` | `feat/rby1-xhand` | Newton sim VR teleop (`sil teleop`) for rby1_xhand, anchored wrist mode |
| hil-serl | `~/workspace/hil-serl` | `bimanual_friday` | `rby1_server.py`, `rby1_reach` experiment, this runbook |
| rby1-sdk | `~/workspace/rby1-sdk` | — | vendor Python SDK (pip `rby1-sdk` 0.10.0), only the follower uses it |

## 2. System map

Two independent stacks. Both start from the same Quest app and the same
`vr_teleop --robot rby1_xhand`.

```
A. VR → Newton sim (visual hands, physics, no ROS)
   Quest (hand mode) ─TCP:8000─▶ vr_teleop (IK: 12 XHand joints/side) ─ZMQ:5555─▶ holiday-lab `sil teleop` (fabrics) ─▶ Newton

B. HIL-SERL → holiday controller (the real-robot stack; SIL = vendor simulator)
   hil-serl env ─HTTP:5010─▶ rby1_server.py ─ROS2 holiday/manipulation─▶ controller_cpp_node (fabrics) ─SHM─▶ rby1-driver ─gRPC─▶ RB-Y1 | rby1-sim
                                            ◀─ROS2 holiday/joint_states─ (+ URDF FK = EE pose)         └─RS485─▶ XHand | loopback
   Quest (controller mode) ─TCP:8000─▶ hil-serl QuestReader (grip = intervene, X = episode end)
```

The Manipulation message is the same one vr_teleop's ROS 2 publisher and the
inference stack emit on every holiday robot: `LinkReaching` per wrist,
`JointReaching` per finger. Stack B therefore moves to the real robot with no
code change (section 6).

## 3. Controls

| Where | Input | Effect |
|---|---|---|
| vr_teleop terminal | `wr` / `wl` | toggle right/left wrist lock (start LOCKED; unlock = anchor latched, no jump) |
| vr_teleop terminal | `fr` / `fl` | toggle finger lock (start FREE) |
| Newton viewer | `t` / `r` / `SPACE` | reset variant / save + next variant / pause |
| Quest, hil-serl | GRIP (hold) | intervene that arm (clutch, anchored) |
| Quest, hil-serl | X | episode end |
| SIL | `./sil.sh enable|disable` | controller rollout on/off (the robot holds its pose when off) |

## 4. One-time setup

```bash
# Quest: developer mode, hand tracking on, Remy2/remy3 app installed; adb on this PC
adb devices

# remy2 vr_teleop venv (py3.12; pin/nlopt/torch-cpu)
cd ~/workspace/remy2/vr_teleop && uv sync

# holiday-lab venv exists (.venv); nothing to add

# holiday SIL images (≈1 h first time; driver + controller, x86, XHand module off)
cd ~/workspace/holiday-rby1/src/drivers/rby1-driver && ./sil.sh build
docker pull rainbowroboticsofficial/rby1-sim:0.10.6-a_v1.2

# hil-serl server image = driver image + flask/scipy/absl (built on first run)
~/workspace/hil-serl/scripts/rby1_server_docker.sh --shell true
```

`sil.sh up` and `sil.sh rviz` open X windows: run from a terminal on this
desktop (`DISPLAY=:1`), not over ssh.

## 5. Every session

### 5.1 Stack A — VR teleop in the Newton sim

```bash
# shell 1: sim (GL window)
cd ~/workspace/holiday-lab
uv run --no-sync sil teleop --scenario rby1_xhand_robot_only \
    --config apps/sil/teleop/configs/rby1_xhand.yaml --streamer-backend none

# shell 2: VR bridge (PC is the TCP server — start it before the app streams)
adb reverse tcp:8000 tcp:8000
cd ~/workspace/remy2/vr_teleop
uv run --no-sync python main.py --robot rby1_xhand           # hands mode, ZMQ :5555
# headset: Remy2 app → TCP (wired), localhost, 8000, hand mode → start streaming
# bridge: with hands visible and arms forward, type wr then wl
```

Teleop config has `br_anchored: true`: on `wr`/`wl` the robot's current
wrist pose and the operator's wrist are latched and the robot follows the
displacement. Locking again drops the anchor. Position box ±0.5 m
(`br_anchor_pos_limit`).

### 5.2 Stack B — HIL-SERL against the controller, SIL

```bash
# 1. stack: vendor sim (2 windows) + driver (loopback XHands) + controller
cd ~/workspace/holiday-rby1/src/drivers/rby1-driver
RBY1_SIL_XHAND=1 ./sil.sh up
./sil.sh enable                       # rollout on
./sil.sh ready                        # spec named state, ~5 s
./sil.sh rviz                         # optional: the only view that shows the XHands

# 2. hil-serl server (container; :5010, ROS domain 138)
cd ~/workspace/hil-serl && scripts/rby1_server_docker.sh
curl -s -X POST http://127.0.0.1:5010/left/getstate | head -c 200   # pose ≈ RESET_POSE

# 3. teleop / data collection (conda hilserl), Quest in CONTROLLER mode
adb reverse tcp:8000 tcp:8000
conda activate hilserl
cd examples/experiments/rby1_reach && python teleop_test.py

# teardown
./sil.sh down; docker rm -f rby1-server-$USER
```

Gate before anything else (same five as the Friday bring-up): zero-policy
hold-still → grip = that arm tracks → release = freeze → re-grip elsewhere =
no jump → headset off mid-grip freezes.

Hands in SIL: `RBY1_SIL_XHAND=1` makes the driver's XHand slots a loopback
(real slew loop, state = command), so `getstate.hand_q` and the RViz hands
follow `POST /<side>/hand {"q": [12]}`. The vendor simulator itself draws the
stock grippers and never the XHands.

## 6. Real robot (not yet done)

Stack B unchanged: replace `sil.sh` by the Orin procedure in the holiday
`src/drivers/rby1-driver/README.md` (`just rt run rby1_xhand`, `torque on`,
`control on`, `pose ready`), keep ROS domain 138, point `scripts/rby1_server_docker.sh`
at the robot's network (CycloneDDS interface). The XHand module builds only
on aarch64 (vendor `.so`), which is why SIL uses the loopback.

## 7. Knobs

| Knob | Where | Value | Meaning |
|---|---|---|---|
| `RESET_POSE` | `rby1_reach/config.py` | FK of `link_<side>_arm_6` at spec `ready` | episode start; recompute if the named state changes |
| `POSE_LIMIT_DELTA` | same | ±0.3 m / ±0.8 rad | box around RESET_POSE |
| `--position_weight/--orientation_weight` | `rby1_server.py` | 5 / 1 | LinkReaching weights |
| `body_reaching.*` | holiday `specs/robot/rby1_xhand/common.controller.yaml` | kp 20, kd 9.2 | wrist BR gains in the real controller |
| `stream.max_step` | driver params (`ros2 param set /rby1_driver stream.max_step`) | 0.3 rad | driver rejects a command further than this from the measured pose |
| `br_anchored`, `br_anchor_pos_limit` | holiday-lab `apps/sil/teleop/configs/rby1_xhand.yaml` | true, 0.5 m | Stack A delta mode |
| `ee.*.orientation_euler` | remy2 `vr_teleop/configs/robots/rby1_xhand.yaml` | FM26B's (placeholder) | wrist orientation correction; shows as a constant twist until the arms-forward probe is redone |
| `human_hand_scale`, `weights.*` | remy2 `retargeting/configs/ik/xhand_right.yaml` | 1.0, prod defaults | XHand IK tuning |

## 8. Hard-won lessons

- **LinkReaching target_frame = `link_<side>_arm_6`**, the controller spec's
  `body_reaching` slot name (wrist flange). `<side>_hand_link` matches no slot
  and is dropped with no log line (`br_by_name.find` → continue). Reference
  `base` → `floating_base`. The XHand base sits 0.1363 m further along −Z.
- **Fingers are non-fabric "held" actuators**: `JointReaching` on
  `<side>_hand_*` bypasses fabrics, goes controller → SHM `/hday_ctrl_cmd` →
  `XHandDriver` (slew at URDF velocity limits × `xhand.velocity_scale`). On
  rollout start they latch to the measured pose.
- **XHand IK frame**: the XHand URDF base (`*_hand_link`: +Z distal, +Y thumb,
  +X palm) already matches the retargeter's wrist-local convention, so
  `xhand_right.yaml` has no `wrist_calibration`. A hand profile must declare
  only the 12 real joints; the 5×4 template helper would put 8 phantom joints
  on the wire.
- **holiday-lab spec robots need glibc ≥ 2.38** (`hday` extension); this host
  has 2.35, so `rby1_xhand` is loaded from `assets/robots/rby1_xhand/robot.urdf`
  (base_env falls back to `robot.urdf` when `robot.usd` is absent).
- **Null-space drift under fabrics**: with the wrist pose held, arm joints keep
  moving for seconds toward the state-attraction targets. Expected; check on
  the robot whether the controller spec needs stronger attraction.
- Both stacks bind :8000 for the Quest; run one bridge at a time.

## 9. File map

| File | Role |
|---|---|
| `serl_robot_infra/robot_servers/rby1_server.py` | HTTP contract ↔ Manipulation API; FK from joint_states; `/hand` |
| `serl_robot_infra/robot_servers/urdf_fk.py` | chain FK shared with friday_server |
| `serl_robot_infra/franka_env/descriptions/rby1_xhand.urdf` | kinematics-only URDF for FK |
| `scripts/rby1_server_docker.sh`, `scripts/Dockerfile.rby1_server` | server container (driver image + flask/scipy/absl) |
| `examples/experiments/rby1_reach/` | experiment config (reset poses, limits), teleop_test |
| remy2 `retargeting/retargeting/profiles/xhand.py`, `configs/ik/xhand_right.yaml`, `assets/xhand_right/` | XHand hand profile + IK |
| remy2 `vr_teleop/configs/robots/rby1_xhand.yaml`, `configs/rby1_xhand.yaml` | wrist/base profile (placeholder), ROS 2 routing |
| remy2 `vr_teleop/tools/rby1_xhand_follower.py` | ZMQ → rby1-sdk direct path (no holiday controller); unverified XHand vendor API |
| holiday-lab `assets/robots/rby1_xhand/`, `assets/scenarios/rby1_xhand_robot_only/`, `apps/sil/teleop/configs/rby1_xhand.yaml` | Newton sim robot + teleop config |
| holiday-lab `policies/structured/teleop.py` (`br_anchored`), `envs/base_env.py` (URDF fallback) | Stack A changes |
| holiday `src/drivers/rby1-driver/sil.sh`, `rby1_driver/src/xhand_driver.cpp`, `docker/rviz/` | SIL: loopback hands, rviz view |
