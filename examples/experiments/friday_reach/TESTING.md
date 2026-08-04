# friday_reach — issue-by-issue test guide

Three issues, tested in order. Each section: what proves the issue done,
exact commands, expected output, and what failure means. Do NOT skip ahead —
issue 3 assumes 1 and 2 pass.

Shells legend:
- **[ROS]** = ROS 2 distro python, robot LAN, `ROS_DOMAIN_ID` set
- **[conda]** = the `hilserl` conda env (python 3.10, jax); never imports rclpy

---

## Issue 1 — HIL-SERL Friday 이식 (bimanual / 허리·손 고정 / 양팔 impedance)

Tests the plumbing: hday topics ↔ friday_server ↔ FridayArmEnv, stereo split,
12D fixed-gripper action space. No headset, no policy, no motion required
until step 1.5.

### 1.1 Server ↔ topics

```bash
# [ROS] confirm the real topic names first
ros2 topic list | grep -E "impedance_target|joint_state|stereo_head"

# [ROS] start the server — FK mode (EE pose from joint_state + URDF; no
# EE-pose topic needed on the robot). Cmd topics default to the canonical
# /hday/friday/impedance_target_* (message_names.h).
python serl_robot_infra/robot_servers/friday_server.py

Expect: two startup lines printing cmd/state topics, Flask on :5010.

### 1.2 State route (per arm)

```bash
curl -s -X POST http://127.0.0.1:5010/left/getstate | python3 -m json.tool
curl -s -X POST http://127.0.0.1:5010/right/getstate | python3 -m json.tool
```

Expect: JSON with `pose` = 7 floats matching
`ros2 topic echo --once <state_topic>` (xyz + quat xyzw). `vel/force/torque/
q/dq` are **zero stubs** for now — the issue text lists them; they become real
when the robot publishes those topics (server-side TODO, one subscription
each).

**503 `{"error": "no EE pose received yet"}`** = no joint_state arriving (FK
mode) or no publisher matched (state-topic mode). Check topic name,
`ROS_DOMAIN_ID`, and QoS (server subscribes BEST_EFFORT sensor-data; a
RELIABLE-only publisher also matches). FK mode also prints a "joint_state
missing [...]" warning if the msg lacks chain joints.

### 1.3 Stereo route + split

```bash
curl -s http://127.0.0.1:5010/getimage -o /tmp/stereo.jpg && xdg-open /tmp/stereo.jpg
```

Expect: one JPEG, both eyes side by side. The env splits at width/2 into
`stereo_left` / `stereo_right` — if the wire format is top-bottom or
different, fix `FridayArmEnv.get_im` before proceeding.

### 1.4 Env spaces, no robot needed

```bash
# [conda]
cd examples/experiments/friday_reach
python -c "
import sys, pathlib; sys.path.insert(0, str(pathlib.Path('..').resolve()))
from experiments.friday_reach.config import TrainConfig
env = TrainConfig().get_environment(fake_env=True, classifier=False)
print('action:', env.action_space.shape)          # (12,)  6D x 2 arms, no gripper
print('obs   :', list(env.observation_space.spaces.keys()))
print('state :', env.observation_space['state'].shape)
"
```

Expect: `action: (12,)`; obs keys `state`, `left/stereo_left`,
`left/stereo_right`; no gripper anywhere. This also proves the learner can
build the env with no robot/network (fake_env path).

### 1.5 Command path (first motion — e-stop in reach)

```bash
# [ROS] watch the command topic
ros2 topic echo /hday/friday/impedance_target_left

# read current pose, add +2 cm z, post it:
curl -s -X POST http://127.0.0.1:5010/left/getstate | python3 -c "
import json,sys; p=json.load(sys.stdin)['pose']; p[2]+=0.02; print(json.dumps({'arr':p}))
" | curl -s -X POST -H 'Content-Type: application/json' -d @- http://127.0.0.1:5010/left/pose
```

Expect: one PoseStamped on the topic; with the impedance controller enabled,
the left EE rises ~2 cm. Repeat for right. This is the entire env→robot
path minus gym.

**Issue 1 done when:** 1.2–1.5 all pass on both arms. Known gaps (tracked, not
blockers): vel/wrench/q/dq stubs; reference limiting is env-side only
(per-step ACTION_SCALE clip + workspace box — controller side should add a
target-vs-current clamp).

---

## Issue 2 — VR teleop ↔ policy switch 데이터 수집 환경

Tests: Quest wrist input, clutch residual mapping, grip = switch,
SpacemouseIntervention-compatible wrapper, interventions land in BOTH buffers.

### 2.0 Math (already automated, no hardware)

```bash
# [conda or system python]
python examples/experiments/friday_reach/test_quest_math.py   # 7 checks pass
```

Covers: LH→RH chirality, base-yaw delta conjugation, jig-offset cancellation,
coupling zero-at-latch / self-healing / steady-saturation.

### 2.1 Quest link alone (no robot)

```bash
# [conda] PC listens; then start the streaming app on the headset,
# controller mode, target = this PC's IP, TCP port 8000
python serl_robot_infra/franka_env/quest_reader.py
```

Expect: "Quest app connected", then per-side lines with pose + grip/trigger
analogs. Move a controller — positions change; squeeze grip — value goes 0→1.
Check axis sanity here: push the controller **forward** (toward the robot) —
the printed delta should be one dominant BASE axis. If forward maps to the
wrong axis, tune `QuestReader(base_yaw_deg=...)` (180 = friday mk3 value from
remy2; gen2 profile uses 90).

### 2.2 Wrapper contract (robot state needed, motion optional)

Covered by issue-3's teleop_test (below) — it prints `INTERVENING` only while
grip is held, which is exactly `info["intervene_action"]` present/absent. The
interface match to SpacemouseIntervention is by construction (same
`(action, replaced)` + info keys); training scripts are untouched.

### 2.3 Interventions reach BOTH buffers (the actual data-collection claim)

Needs actor + learner (actor blocks on `wait_for_server=True`). No real demos
exist yet — make a schema-valid dummy so the learner starts:

```bash
# [conda]
cd examples/experiments/friday_reach
python make_dummy_demo.py                      # -> demo_data/dummy_demo_10.pkl

# shell A (learner, no robot needed — fake env):
bash run_learner.sh --demo_path=demo_data/dummy_demo_10.pkl --debug

# shell B (actor, robot server + headset up; --debug disables wandb):
bash run_actor.sh --debug
```

CAVEAT: `classifier_ckpt/` doesn't exist yet → the reward classifier is
randomly initialized → episodes may end ("success") at random moments and the
robot re-homes on reset. Acceptable in air; don't fight it — this test is
about buffers, not rewards.

Procedure: let the actor run ≥ `buffer_period` (1000) steps (~2 min at 10 Hz).
Hold grip for a few stretches. The tqdm bar plus episode stats show
`intervention_count` / `intervention_steps`. Then:

```bash
# [conda] count what landed where
python3 - <<'EOF'
import glob, pickle
rl   = sum(len(pickle.load(open(f,'rb'))) for f in glob.glob('../../experiments/friday_reach/debug/buffer/*.pkl'))
demo = sum(len(pickle.load(open(f,'rb'))) for f in glob.glob('../../experiments/friday_reach/debug/demo_buffer/*.pkl'))
print(f"RL buffer: {rl}   demo(intervention) buffer: {demo}")
EOF
```

Expect: `demo > 0` and roughly equal to the intervention_steps you held grip
for, `rl >= demo` (every step goes to RL; intervention steps go to both).
Zero demo transitions despite holding grip = wrapper contract broken — stop
and debug before collecting anything real.

**Issue 2 done when:** 2.1 axes verified, 2.3 shows demo-buffer growth only
from grip-held periods.

---

## Twin SIL — full loop against the impedance twin (no robot needed)

The digital twin (mosesnah/holiday `test/mk5-cartimp`) runs the same
Jacobian-transpose Cartesian impedance as the driver
(`docs/arm_impedance.md`) and now publishes EE-pose feedback
(`/hday/friday/{left,right}_ee_pose`, anchor pose, **ROOT frame** — the same
frame `impedance_target_*` consumes, and the same topic names friday_server
expects by default). Run ALL of issue 3 here before hardware.

```bash
# shell 1 — twin (its own venv)
cd /home/user/workspace/mosesnah/holiday/src/digital-twin
uv run ./scripts/run_twin.py -c friday_gen1/alone_v1

# every other ROS shell — INSIDE the twin container. On this workstation the
# host has bare humble (no cyclonedds rmw) while the twin container (jazzy,
# net=host) carries the whole matched DDS stack preconfigured — so run ALL
# ROS shells in the container and keep the host ROS-free. (cyclonedds/domain
# 138/192.168.10.x is the real-robot ops convention; only the container
# follows it here.)
docker exec -it <twin-container-name> bash    # docker ps for the name
source /opt/ros/jazzy/setup.bash
ros2 node list                        # expect /digital_twin_node

# shell 2 — enable impedance + REAL gains (defaults are damping-only = no tracking!)
for S in left right; do
  ros2 param set /digital_twin_node imp_pos_enabled_$S true
  ros2 param set /digital_twin_node imp_rot_enabled_$S true
  ros2 param set /digital_twin_node imp_kp_$S "[800.0, 800.0, 800.0]"
  ros2 param set /digital_twin_node imp_bp_$S "[30.0, 30.0, 30.0]"
  ros2 param set /digital_twin_node imp_kr_$S "[80.0, 80.0, 80.0]"
  ros2 param set /digital_twin_node imp_br_$S "[3.0, 3.0, 3.0]"
done
ros2 topic hz /hday/friday/left_ee_pose        # expect ~50 Hz

# shell 3 — friday_server, INSIDE the container (a copy of the file lives on
# the holiday mount at src/digital-twin/scripts/friday_server.py — re-copy
# from hil-serl/serl_robot_infra/robot_servers/ after edits). flask/absl-py
# must be pip-installed in the container once. net=host, so the conda shell
# reaches :5010 at localhost.
docker exec -it <twin-container-name> bash
source /opt/ros/jazzy/setup.bash
python3 /workspaces/holiday/src/digital-twin/scripts/friday_server.py
# (cmd topics default to /hday/friday/impedance_target_* — message_names.h canon)
# (state topics: defaults already match the twin's ee_pose topics)

# shell 4 — [conda] quest connected, then
cd examples/experiments/friday_reach && python teleop_test.py
```

Twin-specific caveats (from arm_impedance.md tuning notes):

- **Enable BEFORE teleop_test** — reset() streams targets; with impedance off
  they are latched-then-discarded on the enable edge, not tracked.
- **Reset pose (arms straight down) is singular for z** — set RESET_POSE with
  a bent elbow, or jog the arms off singularity first (renderer gizmo:
  `python src/controller/python/ros2/renderer.py --node digital_twin_node`,
  which also draws measured+target EE triads — no extra viz needed).
- **Keep Bp ≲ 30, Br ≲ 3 at 240 Hz** (explicit damping limit-cycles above).
- **Drive PD partial closure**: the twin's drive PD keeps holding streamed q,
  so the EE settles between the two springs — tracking undershoots. Lower arm
  `actuator_gains` in the env config (+ `comp: true`) for hardware-like feel.
- Everything is ROOT-frame consistent end to end: twin feedback (ROOT) →
  clutch coupling → absolute target (ROOT) → twin target (ROOT). The clutch's
  base-yaw probe still applies to the QUEST side only.

Pass criteria = the same five behaviors as issue 3.2 below, but in sim: hold
still on zero policy, grip→track, release→freeze, re-grip→no jump,
staleness-freeze. Also sanity-tune ACTION_SCALE here first (3.3) — expect to
re-tune on hardware where drive-PD closure differs.

---

## Issue 3 — 브링업: zero policy switch 동작

Sanity check in air: robot only moves while grip is held; freezes on release;
action scale tuned by feel. Run the Twin SIL section first — same tests, zero
risk.

One clarification vs the issue text ("zero action policy로 actor 실행"): a
freshly initialized SAC policy is NOT a zero policy — sampling is stochastic
(`argmax=False` even in eval), so a fresh actor jitters. `teleop_test.py` IS
the zero-policy actor loop (literally `env.step(zeros(12))` at 10 Hz), without
RL plumbing. Use it for this issue; the fresh-actor run belongs to issue 2.3.

### 3.0 Preconditions

- Issue 1 passed. Impedance controller enabled, arms in air, e-stop in reach.
- `RESET_POSE` / `ABS_POSE_LIMIT_*` in config.py measured (no `# MEASURE`
  placeholders left) — **reset() DRIVES BOTH ARMS to RESET_POSE.**

### 3.1 Zero policy holds still

```bash
# [conda]
cd examples/experiments/friday_reach
python teleop_test.py     # confirms reset poses, then Enter
```

After the reset move: arms hold. Verify no drift for ~30 s. The console
prints `policy(zero)` each second. `ros2 topic hz` on a command topic shows a
steady 10 Hz of (near-)constant targets — commanded, and still.

Drift here = state topic noise feeding back through the (zero + currpos)
integration; check the EE pose topic stability before blaming the env.

### 3.2 Grip = move, release = freeze

- Hold RIGHT grip, move your hand: right arm follows 1:1 (delta), left arm
  stays. Console flips to `INTERVENING`.
- Release mid-motion: arm freezes within a step or two. Move your hand around
  freely while released: nothing moves (clutch).
- Reposition, re-grip: NO jump (coupling error is zero at the press instant).
- Same for LEFT grip / left arm. Both grips: both arms.
- Kill the headset app while gripping: within 0.25 s the wrapper drops the
  side and the arm freezes (staleness gate) — worth testing on purpose.

### 3.3 Action-scale tuning (직접 움직여보며 확정)

`ACTION_SCALE = (pos_m, rot_rad, _)` per arm in config.py. It is BOTH the
policy's per-step authority and the teleop coupling gain — one knob:

- max chase speed = `pos_m × hz` (0.02 × 10 = 0.2 m/s)
- max rot speed = `rot_rad × hz` (0.1 × 10 = 1.0 rad/s)
- max steady contact force ≈ impedance stiffness × `pos_m`
  (hand penetration saturates at one scale unit)

Procedure: start at (0.02, 0.1). Track a slow figure-8 — smooth? Speed up —
the arm lags then catches up (self-healing; lost motion never accumulates).
If tracking feels sluggish, raise pos_m in 0.005 steps; stop before free-space
overshoot or scary contact force. Then push gently against a fixture: hand
3 cm past the surface should feel like a firm spring, released to level =
force released. Write the final numbers into BOTH arm configs and note them
in the issue.

**Issue 3 done when:** 3.1 no-drift, 3.2 all five behaviors, 3.3 scale
committed to config.py with the measured values.
