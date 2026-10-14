# rby1_reach — HIL-SERL on RB-Y1 + XHand through the holiday controller

Same env/wrappers as `friday_reach`; what changed is the robot server. hil-serl
talks to the holiday controller (fabrics) over the **Manipulation API**, the
wire every holiday robot and vr_teleop use:

```
hil-serl env ──HTTP :5010──▶ robot_servers/rby1_server.py ──ROS 2──▶ controller_cpp_node (fabrics) ──SHM──▶ rby1-driver ──gRPC──▶ RB-Y1 (or vendor sim)
   /<side>/pose     → holiday/manipulation  LinkReaching{target link_<side>_arm_6, ref base, active <side>_arm_0..6}
   /<side>/getstate ← holiday/joint_states  → URDF FK base→link_<side>_arm_6
   /<side>/jointreset → JointReaching = spec `ready` arm joints
   /<side>/hand {"q":[12]} → JointReaching on <side>_hand_* (non-fabric "held" actuators:
                        controller passes them straight to /hday_ctrl_cmd → XHand RS485)
   /getimage          → 503 (no camera on this stack yet)
```

EE frame = `link_<side>_arm_6` (wrist flange). The controller's `body_reaching`
slots are named after that body; a LinkReaching on any other frame (e.g.
`<side>_hand_link`) is dropped silently. The XHand base is 0.1363 m further
along the flange −Z.

## Run (SIL, vendor simulator)

holiday checkout on `feat/rby1-driver` (worktree: `/home/user/workspace/holiday-rby1`):

```bash
cd /home/user/workspace/holiday-rby1/src/drivers/rby1-driver
DISPLAY=:1 ./sil.sh up          # sim (2 windows) + driver + controller; builds images if missing (~1 h)
./sil.sh enable                 # controller rollout on
./sil.sh ready                  # spec named state; ~5 s
./sil.sh watch                  # joints, 2 Hz
./sil.sh down                   # tear down

# server (hil-serl checkout; runs inside the driver image + flask/scipy/absl)
scripts/rby1_server_docker.sh                       # :5010
scripts/rby1_server_docker.sh --shell ros2 topic hz /holiday/joint_states

# env smoke / teleop (conda hilserl)
cd examples/experiments/rby1_reach && python teleop_test.py                 # Quest controller mode, as friday_reach
RBY1_INPUT=hand python teleop_test.py                                        # Quest hand mode via vr_teleop (fingers too)
```

Hand mode needs the bridge: `serl_robot_infra/franka_env/vr_hand_teleop/run_vr_teleop.sh`
(remy2 vr_teleop with the XHand profile), `wr`/`wl` there to free the wrists
(= the deadman), keyboard `x` = episode end. Fingers go to `/<side>/hand`
outside the action. Full procedure and the automatic check: `docs/rby1_runbook.md` §0.

Verified 2026-10-14 in SIL: getstate at `ready` = `RESET_POSE` (FK from URDF),
a +5 cm z LinkReaching reaches in ~1 s with orientation held, `env.reset()` +
`env.step()` move the left wrist and leave the right one still.

## Knobs

`config.py`: `RESET_POSE` = FK of `link_<side>_arm_6` at the spec's `ready`
state (recompute if the named state changes); `POSE_LIMIT_DELTA` ±0.3 m /
±0.8 rad box around it. Reaching gains live in the controller spec
(`specs/robot/rby1_xhand/common.controller.yaml` → `body_reaching`), not here.
Server-side weights: `--position_weight 5 --orientation_weight 1`.

## Not done

- No camera: `image_keys` / classifier need a camera topic + `/getimage`.
- Fingers: operator-driven in hand mode (`XHandPassthrough` → `/<side>/hand`,
  recorded in `info["fingers"]`), not part of the action space. SIL shows them
  through the driver's loopback hands (`RBY1_SIL_XHAND=1`) and `sil.sh rviz`.
- Real robot: same server, `just rt run rby1_xhand` + torque/control on instead
  of `sil.sh`; ROS domain 138 either way.
