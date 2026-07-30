# Friday HIL-SERL Runbook

Operational runbook for the Quest-teleop + RL stack on the friday bimanual
arms. Written 2026-07-30 after twin (SIL) bring-up; the real-robot transition
checklist is at the bottom. Companion docs: `docs/friday_setup_plan.md`
(design), `examples/experiments/friday_reach/TESTING.md` (issue-by-issue
tests), `examples/experiments/friday_reach/README.md`.

## 1. System map

```
Quest app (headset, "Remy2", controller mode)
   │  TCP :8000 (adb reverse, PC is the SERVER — start listener first)
   ▼
QuestReader ─▶ QuestIntervention ─▶ FridayArmEnv x2 ─▶ DualFridayEnv ─▶ RL stack
 (in the teleop/actor process, conda env `hilserl`, NO ROS imports)
   │ HTTP :5000  (/left/... , /right/...)
   ▼
friday_server.py  (ROS env — rclpy + Flask, ONE file)
   │ publish  PoseStamped  cmd_topic_{left,right}     (absolute EE target)
   │ subscribe JointState /hday/friday/joint_state -> URDF FK = EE pose
   │   (FK mode, --fk_urdf; validated vs twin ee_pose: 0.00 mm / 0.000 deg.
   │    Fallback: --state_topic_* PoseStamped subs if a robot ever publishes EE)
   │ subscribe CompressedImage stereo_topic           (head stereo, L|R in one JPEG)
   ▼
J^T Cartesian impedance controller (twin: digital_twin_node / real: robot PC)
```

| Process | Where | Env |
|---|---|---|
| twin (`run-twin -c friday_gen1/alone_v1`) | container (`amazing_cray`, net=host) | container venv |
| `friday_server.py` | wherever ROS + robot topics live (twin: in container, copy at `src/digital-twin/scripts/`) | distro python |
| `teleop_test.py` / actor / learner | host | conda `hilserl` (py3.10, jax) |
| Quest app | headset | — |

Canonical `friday_server.py`: `serl_robot_infra/robot_servers/friday_server.py`.
After editing it, re-copy to the container mount.

## 2. Controls

| Input | Effect |
|---|---|
| GRIP (hold, per side) | intervene that arm — clutch snapshots hand + EE pose at press; position-error coupling while held; release = policy resumes, anchors dropped |
| X (left primary) | episode end (`truncated`, rising edge) → reset drives arms to RESET_POSE |
| A (right primary) | label channel only (`info["right"]`) — free for success marking |
| trigger / B / Y | unused |
| ESC (keyboard) | terminate → episode end |

Clutch math: `action = clip( k·(hand disp − hand₀) − (EE disp − EE₀) ) / ACTION_SCALE`.
Free space = position tracking; contact = hand penetration is a steady force.
Re-grip anywhere = no jump (error is zero at the press instant by construction).

## 3. Knob map (single source of truth)

All in `examples/experiments/friday_reach/config.py` unless noted. Shared
knobs appear ONCE (top of file / `_ArmConfig`) and apply to both arms.

| Knob | Where | Current | Meaning |
|---|---|---|---|
| `TELEOP_EXAGGERATE_POS/ROT` | config.py top | **1.0 / 1.0** | robot displacement = ratio × hand displacement while gripped |
| `POSE_LIMIT_DELTA` | config.py top | ±0.15 m / ±0.4 rad (hardware-safe) | workspace box half-width around each arm's RESET_POSE |
| `ACTION_SCALE` | `_ArmConfig` | (1.0, 0.7854) | policy action space AND teleop speed cap (× hz). Linear-region teleop feel is set by exaggeration, NOT this |
| `RESET_POSE` | per-arm class | measured on twin | episode-start EE pose — MUST be re-measured per robot/init pose |
| `MAX_EPISODE_LENGTH` | `_ArmConfig` | 100000 | episodes end only on X press (bring-up); restore ~120 for RL runs |
| `REALSENSE_CAMERAS` | `LeftArmConfig` | `{}` (twin has no stereo) | restore `{"stereo_left": {}, "stereo_right": {}}` on the real robot |
| impedance gains | `ros2 param set /digital_twin_node imp_*` (twin) / controller side (real) | twin-stable: kp 100, bp 3, kr 3, br 0.1 | wiped on twin restart — re-set every time |
| twin init pose | `digital-twin/configs/friday_gen1/alone_v1.yaml` `init_joint_pos` | arms at URDF limit centers | changing it changes RESET_POSE → re-measure |
| server topics | `friday_server.py` flags | see §4 | remap per deployment |

## 4. Start sequence — twin (SIL)

```bash
# A [container] twin
docker exec -it amazing_cray bash
cd /workspaces/holiday/src/digital-twin && run-twin -c friday_gen1/alone_v1

# B [container] impedance enable + gains — EVERY twin restart (params wipe)
source /opt/ros/jazzy/setup.bash
for S in left right; do
  ros2 param set /digital_twin_node imp_pos_enabled_$S true
  ros2 param set /digital_twin_node imp_rot_enabled_$S true
  ros2 param set /digital_twin_node imp_kp_$S "[100.0, 100.0, 100.0]"
  ros2 param set /digital_twin_node imp_bp_$S "[3.0, 3.0, 3.0]"
  ros2 param set /digital_twin_node imp_kr_$S "[3.0, 3.0, 3.0]"
  ros2 param set /digital_twin_node imp_br_$S "[0.1, 0.1, 0.1]"
done
ros2 topic hz /hday/friday/left_ee_pose        # GATE: ~50 Hz

# C [container] server (twin cmd topics = impedance_target_*)
python3 /workspaces/holiday/src/digital-twin/scripts/friday_server.py
# (cmd topics default to the canonical /hday/friday/impedance_target_* from
#  message_names.h — same names on twin and real robot, no remap)

# D [host] gate, then (re-)measure RESET_POSE if the twin init changed
curl -s -X POST http://127.0.0.1:5000/left/getstate | python3 -m json.tool   # pose, not 503
python3 - <<'EOF'
import requests, numpy as np
from scipy.spatial.transform import Rotation
for side in ("left", "right"):
    p = requests.post(f"http://127.0.0.1:5000/{side}/getstate").json()["pose"]
    e = Rotation.from_quat(p[3:]).as_euler("xyz")
    print(f"{side}: RESET_POSE = np.array([{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}])")
EOF
# -> paste into config.py if changed

# E [host, conda hilserl] teleop
adb reverse tcp:8000 tcp:8000
cd examples/experiments/friday_reach && python teleop_test.py
# THEN start/restart streaming in the Remy2 app (TCP wired, localhost, 8000,
# controller mode) — the listener must exist before the app connects.
```

Restart rules:
- config.py / wrapper / env edits → restart **teleop_test only** (+ reconnect app)
- twin restart → re-run B (params), likely re-measure D, restart teleop_test
- server edit → re-copy to mount, restart C

## 5. Hard-won twin lessons (apply before blaming the teleop stack)

- **Shiver = two references fighting.** Seen twice: (a) renderer jog +
  teleop both commanding one arm — never run both on an arm at once; (b) gain
  sets near the damping ceiling (doc'd bp≲30 @240 Hz) — 800/30 shivered,
  100/3 stable. Amplitude of the flicker ≈ distance between impedance target
  and the drive-PD-held q.
- **Never command measured state.** FridayArmEnv integrates actions on an
  internal SE(3) command (`cmdpos`), seeded from measurement once at startup.
  Zero action = identical command every tick.
- **Soft gains undershoot; the cmdpos integrator + coupling compensate**
  (error keeps integrating until robot displacement matches hand). The force
  ceiling in contact is `kp × lead` — revisit gains and exaggeration together
  for contact tasks.
- **Both processes on one topic**: `ros2 topic info <t> --verbose` is the
  first diagnostic for any "two states" behavior.
- **zsh**: `source /opt/ros/<distro>/setup.zsh` (`.bash` breaks under zsh).
  Host stays ROS-free — all ROS shells live in the container.
- **euler warts** (upstream design, flagged not fixed): `clip_safety_box`
  clips orientation in euler xyz — misbehaves near yaw ±π (left arm yaw is
  0.217 rad from the wrap) and pitch ±π/2. Geodesic cap is the upgrade path.

## 6. REAL ROBOT transition checklist

Do these IN ORDER before first motion. E-stop in reach for every step.

1. **Server flags — FK mode (no EE-pose topic needed on the robot).** The
   server computes EE pose from `/hday/friday/joint_state` + the URDF; the
   robot side needs NO new publishers:
   ```bash
   python serl_robot_infra/robot_servers/friday_server.py \
       --fk_urdf=<path>/friday_gen1_mk5.urdf \
       --stereo_topic=/hday/sensor/camera/stereo_head/image/compressed
   ```
   Chain: `BASE -> waist_1..5 -> {side}_arm_1..7 -> {side}_arm_ee`
   (`--fk_root_link/--fk_ee_link_*` to override; the EE link and root frame
   MUST match what the impedance controller anchors and interprets commands
   in). Joint matching is by name from the JointState msg. Alternative:
   `--state_topic_left/right` if the robot ever publishes EE pose directly.

   **Validate FK against the twin first** (ground truth available): run the
   FK-mode server against the running twin and diff
   `curl :5000/left/getstate` pose vs `ros2 topic echo /hday/friday/left_ee_pose`
   — they must agree to numerical noise. (Twin note: its `ee_pose` is the
   impedance rcc anchor; with rcc offsets at zero that IS the ee body pose.)
2. **`POSE_LIMIT_DELTA`** — already at hardware-safe ±0.15 m / ±0.4 rad
   (done 2026-07-30). Widen deliberately once tracking is proven; 0.4 rad also
   keeps the euler clip box away from its ±pi / gimbal pathologies.
3. **⚠ Revisit `ACTION_SCALE`.** (1.0 m, 0.7854 rad) per step means the
   commanded target can move 1 m per 100 ms tick — fine as an uncapped twin
   speed, dangerous on hardware, and it is the RL policy's per-step authority.
   Paper-style values are 0.01–0.05 m. Teleop feel does NOT need it large
   (exaggeration owns feel); pick the value the POLICY should have.
4. **Re-measure `RESET_POSE`** on the real robot (step D snippet) — twin
   values do not transfer.
5. **Stereo back on**: `LeftArmConfig.REALSENSE_CAMERAS = {"stereo_left": {},
   "stereo_right": {}}`; verify `curl :5000/getimage` shows side-by-side eyes
   and the width/2 split is correct.
6. **Quest base yaw probe** at the real operator spot: push controller toward
   the robot → the coupling error must point robot-forward. Wrong axis →
   `QuestReader(base_yaw_deg=...)` (180 = mk3 profile value; gen2 measured 90).
7. **Impedance gains are the robot side's** — twin-stable values are a sim
   artifact; tune on hardware from soft upward. Reference limiting on the
   controller side (clamp target vs current pose) is still the plan's
   missing safety piece — the env's per-step clip + box is the only limiter.
8. **teleop-only session first** (`teleop_test.py`): hold-still → grip=track →
   release=freeze → re-grip no jump → kill headset mid-grip freezes ≤0.25 s.
9. Then the pipeline in order (TESTING.md §2.3 buffers check, then plan §7):
   classifier data → classifier → demos (X = episode end, A free for success
   marking) → `run_learner.sh --demo_path=...` + `run_actor.sh`.
   Restore `MAX_EPISODE_LENGTH ≈ 120` for RL. Check `classifier_ckpt/` exists
   in the experiment dir for BOTH actor and learner.

## 7. File inventory

| File | Role |
|---|---|
| `serl_robot_infra/franka_env/quest_reader.py` | Quest TCP server, LH→RH + base alignment, poses/buttons |
| `serl_robot_infra/franka_env/envs/quest_intervention.py` | grip clutch + coupling, X episode end, SpacemouseIntervention contract |
| `serl_robot_infra/franka_env/envs/friday_arm_env.py` | per-arm 6D env, cmdpos integrator, stereo-over-HTTP, fake_env offline |
| `serl_robot_infra/robot_servers/friday_server.py` | ROS↔HTTP bridge, per-arm routes, the only ROS process |
| `examples/experiments/friday_reach/config.py` | ALL knobs, arm configs, wrapper chain |
| `examples/experiments/friday_reach/teleop_test.py` | zero-policy bring-up loop |
| `examples/experiments/friday_reach/test_quest_math.py` | 8 math self-checks (`python3 test_quest_math.py`) |
| `examples/experiments/friday_reach/make_dummy_demo.py` | schema-valid demo pkl to unblock the learner |
| `examples/experiments/friday_reach/TESTING.md` | issue-by-issue test procedures |
| `examples/experiments/mappings.py` | registers `friday_reach` (only) |
