# friday_reach — HIL-SERL on the friday bimanual arms

Quest-controller teleop + policy with grip-button intervention switching.
Follows `docs/friday_setup_plan.md`. Action = 12D (2 arms x 6D EE delta,
normalized [-1,1]); fingers and waist are NOT actions — they hold a constant
target owned by the robot side.

## Processes and where they run

| Process | Python env | Why |
|---|---|---|
| `robot_servers/friday_server.py` | ROS 2 distro python | rclpy: pose cmd pub, EE state sub, stereo sub |
| actor / learner / `teleop_test.py` | conda `hilserl` (3.10, jax) | talks to the server over HTTP only |
| Quest streaming app (headset) | — | connects TO the actor machine, TCP :8000 |

The gym env never imports rclpy — the Flask server owns all ROS. Per-arm
routes: `http://127.0.0.1:5000/left/...` and `.../right/...`.

## Topics (flags on friday_server.py)

- publish `PoseStamped` → `/hday/friday/{left,right}_wrist_pose_command`
  (absolute EE target for the jacobian-transpose impedance controller)
- subscribe `PoseStamped` ← `/hday/friday/{left,right}_ee_pose`
  **placeholder — point at the real EE-pose feedback topic**
- subscribe `CompressedImage` ← `/hday/sensor/camera/stereo_head/image/compressed`
  (robot-PC stereo camera; the Quest link carries ONLY controller pose + buttons)

## Intervention model

Grip button (per side) = deadman/clutch. On press both the controller pose and
that arm's EE pose are snapshotted; while held the action is the
position-error coupling

    a = clip( ((hand_p - hand_p0) - (ee_p - ee_p0)) / ACTION_SCALE, -1, 1 )

(same on SO(3) via rotation mismatch → axis-angle). Free space = position
tracking; in contact the hand-past-surface penetration is a steady force
through the impedance controller. Release drops the anchors — reposition the
hand, press again, robot stays put. Any grip held → `intervene=True` for the
step and the transition goes to the demo buffer too. NOTE the flag is
per-step, not per-arm: intervening one arm records the policy's action on the
other arm as part of the "demonstration".

A (right) / X (left) buttons pass through `info["right"]` / `info["left"]`.

## Bring-up order (plan section 7)

```bash
# 0. ROS env, robot LAN
python serl_robot_infra/robot_servers/friday_server.py   # check state topics first!

# 1. teleop only — verify clutch, impedance tracking, workspace clamps
cd examples/experiments/friday_reach && python teleop_test.py

# 2. classifier data + training
python ../../record_success_fail.py --exp_name friday_reach
python ../../train_reward_classifier.py --exp_name friday_reach

# 3. demos (20-30)
python ../../record_demos.py --exp_name friday_reach

# 4. RL with interventions (two shells, from THIS directory)
bash run_learner.sh --demo_path=demo_data/friday_reach_20_demos_<timestamp>.pkl
bash run_actor.sh
```

## Before first motion — measure and fill (marked `# MEASURE` in config.py)

- `RESET_POSE`, `ABS_POSE_LIMIT_LOW/HIGH` per arm (BASE frame, xyz + euler xyz)
- `IMAGE_CROP` once the scene is framed
- Quest base yaw: `QuestReader(base_yaw_deg=...)` — 180 is the friday mk3
  value from remy2; verify with an arms-forward probe (push hand forward,
  robot EE error must point forward in BASE)

## Known simplifications

- `vel/force/torque/q/dq` in getstate are zero stubs until real feedback
  topics exist; `update_param`/`jointreset`/`clearerr` are no-op routes
  (gains are owned by the impedance controller).
- Reset flows through pose targets (`interpolate_move` linspaces quat
  components raw — fine for small moves; keep RESET_POSE near the workspace).
- The base env's reference limiting (paper's safety piece) is the per-step
  ACTION_SCALE clip + `clip_safety_box` workspace box. The controller side
  should ALSO clamp target-vs-current distance (plan section 3) — the reset
  path (`interpolate_move`) bypasses even the workspace clip.
- Success/abort marking is still keyboard-driven (SPACE in
  record_success_fail, ESC for episode end) while both hands hold
  controllers — awkward. A (right) / X (left) are already surfaced in
  `info["right"]`/`info["left"]` but no stock script consumes them; binding
  success to A is a 5-line change in record_success_fail.py when it hurts.
- Plan section 3 wants twist + wrench proprio; both are zero stubs until the
  robot publishes real feedback (proprio_keys currently pose-only).
