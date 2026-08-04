# Friday HIL-SERL Runbook

Operational runbook for the Quest-teleop + RL stack on the **real friday
robot**. The digital twin is an optional SIL sandbox (§6) — useful for
rehearsing without hardware, never required. Companion docs:
`docs/friday_setup_plan.md` (design), `examples/experiments/friday_reach/
TESTING.md` (issue-by-issue tests), `examples/experiments/friday_reach/README.md`.

## 1. System map

```
Quest app (headset, "Remy2", controller mode)
   │  TCP :8000 (adb reverse, PC is the SERVER — start listener first)
   ▼
QuestReader ─▶ QuestIntervention ─▶ FridayArmEnv x2 ─▶ DualFridayEnv ─▶ RL stack
 (teleop/actor process, conda env `hilserl`, NO ROS imports)
   │ HTTP :5010 (/left/... , /right/...)
   ▼
friday_server.py  (ROS: rclpy + Flask, ONE self-contained script)
   │ publish  PoseStamped  /hday/friday/impedance_target_{left,right}   (canon)
   │ subscribe JointState  /hday/friday/joint_state -> URDF FK = EE pose
   │   (FK mode, URDF bundled in-repo; validated vs twin: 0.00 mm / 0.000 deg.
   │    Fallback: --state_topic_* PoseStamped subs)
   │ subscribe CompressedImage /hday/sensor/camera/stereo_head/image/compressed
   ▼
J^T Cartesian impedance controller (real driver — or the twin, §6)
```

| Process | Where (real robot) | Env |
|---|---|---|
| `friday_server.py` | ops PC (jazzy on host) — run STRAIGHT from this repo, no container, no copy. Or the onboard Jetson (§4.6) | distro python + `pip install flask absl-py scipy` |
| `teleop_test.py` / actor / learner | ops PC or any host reaching :5010; teleop/data-collection also runs all-on-Jetson (§4.6) | conda `hilserl` (py3.10, jax) — on the Jetson: repo-local `.venv`, `./scripts/setup_venv.sh` (§4.6) |
| Quest app | headset, USB to the teleop machine | — |
| impedance controller | robot side (already deployed) | — |

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

| Knob | Where | Hardware-safe committed value | Meaning |
|---|---|---|---|
| `TELEOP_EXAGGERATE_POS/ROT` | config.py top | 1.0 / 1.0 | robot displacement = ratio × hand displacement while gripped |
| `POSE_LIMIT_DELTA` | config.py top | ±0.15 m / ±0.4 rad | workspace box half-width around each arm's RESET_POSE; widen deliberately. 0.4 rad also keeps the euler clip box away from its ±π / gimbal pathologies |
| `ACTION_SCALE` | `_ArmConfig` | ⚠ still twin value (1.0, 0.7854) — pick a policy-sane value (paper: 0.01–0.05 m) before RL | policy action space AND teleop speed cap (× hz). Teleop FEEL is set by exaggeration, not this |
| `RESET_POSE` | per-arm class | measure on the robot (§4.3) | episode-start EE pose |
| `MAX_EPISODE_LENGTH` | `_ArmConfig` | 100000 (X-press-only episodes) | restore ~120 for RL runs |
| `REALSENSE_CAMERAS` | `LeftArmConfig` | restore `{"stereo_left": {}, "stereo_right": {}}` on the robot | head stereo via server `/getimage` |
| impedance gains | robot driver side | tune from soft upward | NOT owned by this stack |
| server topics | `friday_server.py` flags | defaults = message_names.h canon | remap only if a deployment diverges |

## 4. Real robot — pre-flight, then start

### 4.1 Pre-flight checklist (e-stop in reach throughout)

1. `ros2 topic list | grep -E "impedance_target|joint_state|stereo_head"` —
   canon topics present.
2. `POSE_LIMIT_DELTA` at ±0.15/±0.4 (committed default). Widen only after
   tracking is proven.
3. **⚠ `ACTION_SCALE`** — decide the policy-sane value now (it is the RL
   action space; teleop does not need it large).
4. Stereo restored in `LeftArmConfig` + `curl :5010/getimage` shows
   side-by-side eyes (env splits at width/2).
5. Quest base-yaw probe at the operator spot: push controller toward the
   robot → coupling error must point robot-forward. Wrong axis →
   `QuestReader(base_yaw_deg=...)` (180 = mk3; gen2 measured 90).
6. Impedance gains: robot side, from soft upward. Reference limiting
   (clamp target vs current pose) on the controller side is still the plan's
   missing safety piece — the env's per-step clip + box is the only limiter.

### 4.2 Start the server (ops PC, jazzy host — no container)

```bash
pip install flask absl-py scipy          # once
source /opt/ros/jazzy/setup.bash         # setup.zsh in zsh
python3 serl_robot_infra/robot_servers/friday_server.py
```

The URDF is bundled at `serl_robot_infra/franka_env/descriptions/` (kinematics
only — mesh refs are never read, so no mesh files are needed); `--fk_urdf`
overrides it, `--fk_urdf=""` falls back to `--state_topic_*` subscriptions.

FK chain: `BASE -> waist_1..5 -> {side}_arm_1..7 -> {side}_arm_ee`
(`--fk_root_link/--fk_ee_link_*` to override — EE link and root frame MUST
match what the impedance controller anchors and interprets commands in).
Joint matching is by name; no EE-pose topic is needed on the robot.

Gate: `curl -s -X POST http://127.0.0.1:5010/left/getstate` returns a pose
(503 = joint_state not arriving; the server also warns if chain joints are
missing from the msg).

### 4.3 Measure RESET_POSE (robot at its rest/start posture)

```bash
python3 - <<'EOF'
import requests, numpy as np
from scipy.spatial.transform import Rotation
for side in ("left", "right"):
    p = requests.post(f"http://127.0.0.1:5010/{side}/getstate").json()["pose"]
    e = Rotation.from_quat(p[3:]).as_euler("xyz")
    print(f"{side}: RESET_POSE = np.array([{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}])")
EOF
# -> paste into config.py (twin values do NOT transfer)
```

Reset-to-where-it-stands = zero motion on the first reset.

### 4.4 Teleop-only session (bring-up gate)

```bash
adb reverse tcp:8000 tcp:8000
conda activate hilserl
cd examples/experiments/friday_reach && python teleop_test.py
# then start streaming in the Remy2 app: TCP wired, localhost, 8000, controller mode
```

Pass = all five: zero-policy hold-still → grip = that arm tracks → release =
freeze mid-motion → re-grip elsewhere = no jump → headset killed mid-grip
freezes ≤ 0.25 s.

### 4.5 Pipeline (plan §7)

classifier data → classifier → demos → RL:

```bash
python ../../record_success_fail.py --exp_name friday_reach
python ../../train_reward_classifier.py --exp_name friday_reach
python ../../record_demos.py --exp_name friday_reach          # X = episode end
bash run_learner.sh --demo_path=demo_data/<file>.pkl          # shell 1
bash run_actor.sh                                             # shell 2
```

Restore `MAX_EPISODE_LENGTH ≈ 120` for RL. `classifier_ckpt/` must exist in
the experiment dir for BOTH actor and learner. Buffer sanity check:
TESTING.md §2.3 (interventions land in RL + demo buffers).

⚠ **`record_success_fail.py` is not headless-ready.** It imports pynput at
module level and toggles the success label with the **spacebar** — over ssh
with no X, the import itself fails. §2 reserves the Quest **A** button as the
label channel (`info["right"]`) for exactly this, but nothing is wired to it
yet. Wire A → `success_key` before running the classifier-data step on the
Jetson (`teleop_test.py` / `record_demos.py` are unaffected — neither imports
pynput).

Restart rules: config/wrapper/env edits → restart teleop/actor only (+
reconnect app). Server edit → restart server.

### 4.6 Variant — everything on the onboard Jetson (headless)

Teleop bring-up and data collection (`teleop_test`, `record_success_fail`,
`record_demos`) do NO policy inference — the loop is Quest TCP + HTTP +
numpy/scipy at 10 Hz. All of it runs on the headless Jetson over ssh; the
whole stack is display-free (the ESC listener disables itself without X —
Quest X button is the episode-end control).

One non-obvious requirement: `config.py` IMPORTS jax/flax/serl_launcher at
module level even though teleop never computes with them — plain CPU jax
(aarch64 wheel) satisfies the import.

Install (validated on the friday Jetson, 2026-08-04: Ubuntu 24.04 aarch64, no
ROS on the host, no conda, python3.12 only). This Jetson is shared — the
install writes **nothing** outside the repo: no apt, no sudo, no `~/.local`,
no `~/.cache`. `.venv/` (855 MB) and `.tools/` (176 MB, holds uv + a private
CPython 3.10 + caches + the server's pip target) are gitignored; deleting
those two dirs is a complete uninstall.

```bash
./scripts/setup_venv.sh              # once — uv + .venv + deps + smoke test
source .venv/bin/activate
```

Do NOT set `PYNPUT_BACKEND=dummy`. Headless, `from pynput import keyboard`
raises ImportError, the guard in `friday_arm_env.py` catches it and prints
"ESC listener disabled" — the designed path. The dummy backend defeats that:
import and `listener.start()` both succeed and the `NotImplementedError`
surfaces in the listener's own thread, where the guard cannot see it, so you
get one spurious traceback per arm. Harmless, but noise. `teleop_test.py` and
`record_demos.py` never touch pynput; only `record_success_fail.py` imports it
at module level (§4.5 caveat).

Deps are pinned in the script, NOT taken from the two `setup.py` files
(they are installed `--no-deps`): upstream's `install_requires` pulls
pymodbus / hidapi / rospkg / tensorflow plus the bogus PyPI `typing` and `zmq`
shim packages, none of which the friday stack needs. `pyrealsense2` and
`easyhid` ARE installed — unused by friday, but `wrappers.py`/`franka_env.py`
import them at module level. jax resolves to the CPU aarch64 wheel (0.6.2),
which is all teleop needs.

**ROS lives in a container, not on the Jetson.** There is no `/opt/ros` here,
so `friday_server.py` (the only rclpy process) borrows the jazzy image already
on the box. The wrapper runs a throwaway `--rm` container as your uid with the
repo bind-mounted, so nothing persists and the long-running `holiday-iekf`
container is left alone. DDS settings are mirrored from it (`ROS_DOMAIN_ID=138`,
cyclonedds, thort profile) — without them the robot's topics are invisible.

**Host vs container split.** Container = ROS only (`friday_server.py`, `ros2`
CLI). Host = everything else: `.venv`, teleop, `adb`, the Quest. `adb` is
already installed on the Jetson host — never `apt install` inside the
container (it is unprivileged and `--rm`, so it would fail and then vanish).
Never `source .venv/bin/activate` inside the container either: the venv is on
the mount, so it is visible, but it puts python3.10 ahead of the image's
python3.12 and rclpy stops importing.

Container names: the server runs as `friday-server-$USER` (image
`cr.hday.io/rtabmap/rtabmap-jazzy:v0.1`). `holiday-iekf` is somebody else's
long-running container and is never touched. For ROS CLI work (`ros2 topic
list/echo/hz`) use a second, independent container — do NOT exec into the
server one, a crashed shell there takes the server with it:

```bash
./scripts/friday_server_docker.sh --shell                 # interactive, cwd=/repo
./scripts/friday_server_docker.sh --shell ros2 topic list # one-shot
docker logs -f friday-server-$USER                        # server output
```

**Port note: the stack is on 5010, not 5000.** Port 5000 is held by the zot
registry container (`*:5000`) on this Jetson, so 5010 is now the committed
default in all three places — `friday_server.py --flask_port`, and
`SERVER_URL` for both arms in `config.py`. Nothing needs a flag; override with
`--flask_port=` + matching `SERVER_URL` if you ever move it.

```bash
# shell 1 — server (robot topics are local to the Jetson: ideal placement)
./scripts/friday_server_docker.sh
# gate: curl -s -X POST http://127.0.0.1:5010/left/getstate  -> real pose
#       (verified 2026-08-04 against the live robot, FK mode, both arms)

# shell 2 — Quest USB plugged into the JETSON
source .venv/bin/activate
adb reverse tcp:8000 tcp:8000        # first time: accept the headset's
                                     # "Allow USB debugging" prompt; udev rule
                                     # (vendor 2833) if adb says no permissions
cd examples/experiments/friday_reach && python teleop_test.py
# headset: Remy2 app, TCP wired, localhost:8000, controller mode
```

`SERVER_URL` stays 127.0.0.1 (server and env on the same Jetson) — only the
port changes.

What does NOT belong on the Jetson: the learner (needs the GPU workstation)
and the actor with a real policy (jax-CUDA has no official aarch64 wheels;
CPU inference of the ResNet encoder is marginal at 10 Hz). For RL, split:
Jetson runs `friday_server` only; workstation runs actor+learner with
`SERVER_URL = "http://<jetson-ip>:5010/<side>/"` in config.py and the Quest
USB moved to the workstation.

## 5. Hard-won lessons (mostly learned on the twin, apply everywhere)

- **Shiver = two references fighting.** Seen as: (a) renderer jog + teleop
  both commanding one arm — never both on an arm at once; (b) gains near the
  damping ceiling. Amplitude ≈ distance between impedance target and the
  drive-held reference. First diagnostic for any "two states":
  `ros2 topic info <topic> --verbose` (publisher count).
- **Never command measured state.** FridayArmEnv integrates actions on an
  internal SE(3) command (`cmdpos`), seeded from measurement once at startup.
  Zero action = identical command every tick.
- **Soft gains undershoot; the cmdpos integrator + coupling compensate**
  (error keeps integrating until robot displacement matches hand). Contact
  force ceiling is `kp × lead` — revisit gains and exaggeration together for
  contact tasks.
- **zsh**: `source .../setup.zsh`, not `.bash`.
- **euler warts** (upstream design, flagged not fixed): `clip_safety_box`
  clips orientation in euler xyz — misbehaves near yaw ±π and pitch ±π/2.
  Geodesic cap is the upgrade path if it ever bites.

## 6. OPTION — digital twin (SIL rehearsal, no hardware)

Same stack, twin instead of robot. Only worthwhile for rehearsing procedures
or debugging the teleop chain; skip when the robot is available.

This test workstation's quirk: host ROS is humble without cyclonedds, so ALL
ROS (twin, params, server) runs inside the twin container (`amazing_cray`,
net=host, jazzy); host stays ROS-free. `friday_server.py` runs from the copy
on the mount (`src/digital-twin/scripts/` — re-copy after edits).

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
ros2 topic hz /hday/friday/left_ee_pose        # ~50 Hz gate

# C [container] server (same command as real; twin consumes the same canon topics)
python3 /workspaces/holiday/src/digital-twin/scripts/friday_server.py

# D [host] measure RESET_POSE for the twin posture (§4.3 snippet), then
#   conda + adb reverse + teleop_test.py exactly as §4.4
```

Twin-only notes:
- Stable twin gains: kp 100, bp 3, kr 3, br 0.1 (800/30/80/3 shivers). These
  are a sim artifact — do NOT carry to hardware.
- Twin has no head stereo (alone_v1) → `REALSENSE_CAMERAS = {}`.
- For teleop feel on the twin, temporarily widen `POSE_LIMIT_DELTA` /
  raise exaggeration — restore with `git checkout -- config.py` after.
- Twin init pose: `digital-twin/configs/friday_gen1/alone_v1.yaml`
  `init_joint_pos` (arms currently at URDF limit centers). Changing it
  changes RESET_POSE → re-measure.
- Renderer (`renderer.py --node digital_twin_node`) and teleop must never
  command the same arm simultaneously.
- Twin's own EE feedback topics (`/hday/friday/{left,right}_ee_pose`, 50 Hz)
  exist and can replace FK mode (`--state_topic_*`) — FK was validated
  against them at 0.00 mm / 0.000°.

## 7. File inventory

| File | Role |
|---|---|
| `serl_robot_infra/franka_env/quest_reader.py` | Quest TCP server, LH→RH + base alignment, poses/buttons |
| `serl_robot_infra/franka_env/envs/quest_intervention.py` | grip clutch + coupling + exaggeration, X episode end, SpacemouseIntervention contract |
| `serl_robot_infra/franka_env/envs/friday_arm_env.py` | per-arm 6D env, cmdpos integrator, stereo-over-HTTP, fake_env offline |
| `serl_robot_infra/robot_servers/friday_server.py` | ROS↔HTTP bridge, FK mode, per-arm routes — the only ROS process |
| `examples/experiments/friday_reach/config.py` | ALL knobs, arm configs, wrapper chain |
| `examples/experiments/friday_reach/teleop_test.py` | zero-policy bring-up loop |
| `examples/experiments/friday_reach/test_quest_math.py` | 8 math self-checks (`python3 test_quest_math.py`) |
| `examples/experiments/friday_reach/make_dummy_demo.py` | schema-valid demo pkl to unblock the learner |
| `examples/experiments/friday_reach/TESTING.md` | issue-by-issue test procedures |
| `examples/experiments/mappings.py` | registers `friday_reach` (only) |
