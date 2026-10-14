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
| hil-serl | `~/workspace/hil-serl` | `bimanual_friday` | everything rby1-specific on our side: `rby1_server.py`, `rby1_reach`, the XHand hand profile + vr_teleop launcher (`franka_env/vr_hand_teleop/`), this runbook |
| holiday (monorepo) | `~/workspace/holiday-rby1` (worktree) | `feat/rby1-driver` + `feat/rby1-driver-sil-hands` | RB-Y1 driver, holiday controller (fabrics) for `rby1_xhand`, SIL (`sil.sh`), specs, URDF. Our one commit: loopback hands + rviz for SIL |
| remy2 | `~/workspace/remy2` | `main`, **read-only**, pinned (f260792f) | vr_teleop + retargeting, used as a library/process by `vr_hand_teleop/run_vr_teleop.sh`; no rby1 code in it |

Not needed: holiday-lab (Newton sim). A `feat/rby1-xhand` branch there has a
working `sil teleop` for rby1_xhand (URDF robot, anchored wrist mode); kept
as a visual/physics sandbox for the hands, not part of this pipeline.

## 2. System map

```
Robot side (SIL = vendor simulator; the robot later, same wire):
   rby1_server.py ─ROS2 holiday/manipulation─▶ controller_cpp_node (fabrics) ─SHM─▶ rby1-driver ─gRPC─▶ RB-Y1 | rby1-sim
                  ◀─ROS2 holiday/joint_states─ (+ URDF FK = EE pose)                        └─RS485─▶ XHand | loopback (SIL)

Operator side, two input modes (one at a time; both bind :8000):
   controller mode: Quest ─TCP─▶ hil-serl QuestReader ─▶ QuestIntervention (grip = intervene, X = episode end) ─▶ env ─HTTP:5010─▶ rby1_server
   hand mode:       Quest ─TCP─▶ vr_teleop (remy2 venv, run_vr_teleop.sh: wrist + 12 XHand joints/side) ─ZMQ:5555─▶ hil-serl [ZmqQuestReader, TODO] ─▶ env ─HTTP─▶ rby1_server (/pose, /hand)
```

`Manipulation` = `LinkReaching` per wrist + `JointReaching` per finger, the
message vr_teleop's ROS 2 publisher and the inference stack emit on every
holiday robot. Nothing changes between SIL and the robot (section 6).

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

### 5.1 Hand-tracking bridge (fingers): remy2 vr_teleop via our launcher

```bash
adb reverse tcp:8000 tcp:8000
cd ~/workspace/hil-serl
serl_robot_infra/franka_env/vr_hand_teleop/run_vr_teleop.sh         # ZMQ :5555
# headset: Remy2 app → TCP (wired), localhost, 8000, hand mode → start streaming
# bridge terminal: hands visible, arms forward → wr, then wl
```

The launcher runs remy2's own `main.py` in remy2's venv after injecting the
XHand hand profile and pointing `--robot` at `rby1_xhand.robot.yaml`,
`--ik-config` at `xhand_right.yaml`, `--config` at `rby1_xhand.routing.yaml`
(all in `vr_hand_teleop/`). Output per side: `arm` {position, rotation wxyz}
in base frame (absent while locked) and `hand.joint_angles` = the 12 XHand
joints by URDF name. `--publisher ros2` sends the same as `holiday/manipulation`
(needs friday_msgs in the environment, i.e. a container). Consuming this in the
hil-serl env (`ZmqQuestReader`) is the open item in section 10.

### 5.2 HIL-SERL against the controller, SIL

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

Fingers from the env: `POST /<side>/hand {"q": [12]}` (XHand SDK order),
echoed back as `getstate.hand_q`.

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
| `ee.*.orientation_euler` | `vr_hand_teleop/rby1_xhand.robot.yaml` | FM26B's (placeholder) | wrist orientation correction; shows as a constant twist until the arms-forward probe is redone |
| `human_hand_scale`, `weights.*` | `vr_hand_teleop/xhand_right.yaml` | 1.0, prod defaults | XHand IK tuning |

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
- **remy2 stays untouched**: hand profiles and `--robot` are closed registries
  in remy2, so `run_vr_teleop.py` injects the profile and replaces
  `resolve_robot_profile` at runtime instead of forking. Pin remy2; the
  launcher checks the three seams it uses and exits if they move.
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
| `serl_robot_infra/franka_env/vr_hand_teleop/` | XHand hand profile (`xhand_profile.py`), IK config + URDF, wrist/base profile (placeholder), routing yaml, `run_vr_teleop.{py,sh}` launcher over remy2 |
| holiday `src/drivers/rby1-driver/sil.sh`, `rby1_driver/src/xhand_driver.cpp`, `docker/rviz/` | SIL: loopback hands, rviz view |

## 10. Open items

1. `ZmqQuestReader` + intervention wrapper for hand mode: consume the vr_teleop
   wire (wrist pose already in base frame; `arm` absent = that side released,
   which is the deadman; episode end from the keyboard), forward the 12 finger
   targets to `/<side>/hand`. Decide later whether fingers enter the action
   space (12D → 36D) or stay operator-driven.
2. Camera on the rby1_xhand stack → `/getimage`, `image_keys`.
3. Arms-forward probe on the robot → `rby1_xhand.robot.yaml` orientation values.
