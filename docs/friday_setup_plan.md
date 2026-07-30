# HIL-SERL on Friday: Setup Plan

Plan for running HIL-SERL (Luo et al., 2024, [paper](../hil-serl-paper.pdf)) on the Friday humanoid.
Written 2026-07-28.

## 1. Paper summary (what the system needs)

- **Algorithm**: RLPD (off-policy, SAC-based) + 20-30 human demos + live human corrections.
  Actor and learner run async (agentlace); learner samples 50/50 from demo buffer and RL buffer.
  Interventions go to both buffers; the policy's own transitions only to the RL buffer.
- **Reward**: binary success classifier on images. ~200 positive / ~1000 negative frames
  (~10 teleop trajectories, ~5 min collection). Target >95% accuracy.
- **Vision**: frozen ImageNet-pretrained ResNet-10, per-task image crops resized to 128x128.
  Wrist cams preferred, side cams as needed.
- **Proprio / actions**: EE pose expressed relative to the episode-start frame (ego-centric —
  this is what buys spatial generalization). Policy outputs per-step 6D EE delta to a Cartesian
  impedance controller with **reference limiting** (clamp target near current pose) for contact
  safety. Gripper handled by a separate DQN grasp critic over discrete actions, with a small
  penalty on gripper actions.
- **Results**: 100% success on all tasks, 1-2.5 h real-world training, single RTX 4090.
- **Training recipe**: pick cameras → collect classifier data → train classifier → record demos
  → RL training with interventions until convergence. Avoid long interventions that end in
  success early in training (causes value overestimation, unstable dynamics).

## 2. Friday hardware mapping

| Component | Paper | Friday |
| --- | --- | --- |
| Robot | Franka Panda x1-2 | Bimanual 7DoF arms (waist/mobile/head locked) |
| Controller | Cartesian impedance + reference limiting | Existing Cartesian impedance controller; add reference limiting + workspace box |
| Gripper | Franka / Robotiq | None — no gripper/hand action in RL |
| Teleop | SpaceMouse | Oculus Quest controller (remy2 vr_teleop), clutch + grip-button deadman |
| Cameras | Wrist RealSense + side cams | Head stereo (fixed side view) |
| GPU | RTX 4090 | RTX 5080 |

## 3. Robot infra (`serl_robot_infra`)

New `friday_env` mirroring `franka_env` (reuse the Holly ROS2 server pattern — see §8):

- **Server**: process exposing per-arm EE pose target (impedance setpoint), EE state
  (pose, twist, wrench if force sensing exists).
- **Controller safety**: reference limiting — clamp commanded target within a small ball around
  the current EE pose every step (the paper's key safety piece for RL exploration in contact).
  Plus a workspace bounding box (`ABS_POSE_LIMIT_*` pattern already in configs). Soft impedance
  gains for contact tasks. Nullspace posture hold for the elbow (arms are 7DoF).
- **Action space**: per-arm 6D delta pose in the current EE frame, clipped per step
  (`ACTION_SCALE`). Bimanual = 12D continuous. Start with single-arm tasks.
- **Proprio obs**: EE pose relative to episode-start frame (`RelativeFrame` wrapper does this),
  twist, wrench.
- **Gripper / hand**: NOT handled — no gripper action in the RL problem (decision 2026-07-29).
  Use `setup_mode = "*-fixed-gripper"` (objects pre-grasped or task needs no grasp). No grasp
  critic, no `GripperPenaltyWrapper`. Action per arm = 6D pose residual only.

## 4. Teleop (Oculus Quest controller via remy2)

Wrist SE(3) pose from the Quest controller, using `/home/user/workspace/remy2` `vr_teleop`
(`VRAppControllerInterface` in `vr_app_controller_tracker.py`): Unity stream → controller
anchor → wrist-frame correction → robot frame via `compute_wrist_pose`. Exposes per-side
pose + trigger/grip analogs + A/B/X/Y buttons. Design points:

- **Residual/clutch interface, not absolute.** The policy action space is relative deltas, and
  interventions must live in the same space. Clutch-engaged relative mapping: on engage,
  snapshot controller pose + robot EE pose; commanded delta = current controller pose −
  snapshot, converted to per-step clipped deltas. Absolute mapping jumps on engage and fights
  the policy.
- **Intervention switch = grip button.** Grip held → human action overrides policy and
  `intervene=True`; released → policy resumes. Implement as a wrapper replacing
  `SpacemouseIntervention` with the same output contract (`action`, `intervene_action` in
  info), so training scripts stay untouched.

Same rig serves all three data needs: classifier data, demos, live corrections.

## 5. Cameras

- Head stereo via ROS topic `/hday/sensor/camera/stereo_head/image/compressed` (CompressedImage,
  both eyes in one frame). New capture class subscribes, decodes, splits left/right eye into two
  named streams (`stereo_left`, `stereo_right`) replacing `RSCapture`.
- Head fixed = side-camera geometry. No wrist cam; compensate with tight per-task crops
  (`IMAGE_CROP`) → 128x128. Depth unused.
- Known gap vs paper: no wrist cam. If precision insertion underperforms, add a cheap wrist cam.

## 6. GPU (RTX 5080)

Blackwell = sm_120. README's pinned `jax[cuda12_pip]==0.4.35` (Oct 2024) predates Blackwell.
Install recent jax (`pip install -U "jax[cuda12]"`, needs CUDA 12.8+ driver), then fix any API
drift in `serl_launcher`. Verify: `python -c "import jax; print(jax.devices())"` and one dummy
RLPD update step. 16 GB VRAM is plenty (paper used a 24 GB 4090; networks are small).

## 7. Bring-up order

1. Server + `FridayEnv` teleop-only: Quest controller clutch drives arm; verify impedance
   tracking, reference limiting, workspace clamps.
2. Reward classifier: `examples/record_success_fail.py` → ~200 pos / ~1000 neg →
   `examples/train_reward_classifier.py`, check >95% accuracy.
3. Demos: `examples/record_demos.py`, 20-30 trajectories.
4. RL: `examples/train_rlpd.py` actor + learner, intervene via deadman. Expect 1-2.5 h.
   Avoid long success-ending interventions early in training.

First task: single-arm, short-horizon (pickup or peg-in-hole) before anything bimanual.
Per task, copy `examples/experiments/holly_pickup_insertion/` as scaffolding and register in
`examples/experiments/mappings.py`.

**Biggest risks, in order**: Quest-controller clutch/deadman ergonomics (bad teleop → bad demos → nothing
works), no wrist cam for precision tasks, jax-on-Blackwell version dance.

## 8. Fork changes vs upstream (`8f1daa0`)

This repo forked from rail-berkeley/hil-serl at `8f1daa0`. Colleague's changes since:

- **ROS1 → ROS2 port of the robot servers** (`serl_robot_infra/robot_servers/franka_server.py`,
  `franka_gripper_server.py`): `rospy` → `rclpy` (node spun in a background thread next to
  Flask). Retargeted to the in-house "hday" FR3 stack:
  - EE command: publish `PoseStamped` to `/hday/fr3_controller/desired_cartesian_pose`
  - State: subscribe `/hday/rt_franka/ef_pose`, `/hday/rt_franka/joint_state`,
    `/hday/sensor/ft_sensor` (external F/T sensor instead of Franka's estimated wrench)
  - Gains: `ControllerGain` service (`hday_controller_msgs`) replaces dynamic_reconfigure;
    nullspace gains currently hardcoded (marked TODO)
  - Joint reset: motion-planner `Move` service (`hday_motion_planner_msgs`) instead of
    launching/killing a joint controller; controller reset via `Trigger` services
  - Removed: roslaunch impedance start/stop, error recovery (`/clearerr` route deleted)
  - Gripper: `franka_gripper` ROS1 action goals → `franka_msgs` ROS2 `ActionClient`
    (`/fr3_gripper/move`, `/fr3_gripper/grasp`)
- **New experiment `examples/experiments/holly_pickup_insertion/`**: `HollyEnv` (subclass of
  `FrankaEnv`) with single RealSense side cam shared between policy and classifier keys,
  keyboard-'m'-triggered regrasp routine, MP-based reset, no `interpolate_move` in reset
  (removed in `47f7f11`); `GripperPenaltyWrapper`; config with crops, compliance/precision
  gain sets, pose limits; registered in `mappings.py`; `run_actor.sh` / `run_learner.sh`.
- **`examples/record_success_fail.py`**: space bar now toggles success state (sticky until
  episode end) instead of marking a single frame; prints running counts.
- Misc: `.gitignore` added (pyc, egg-info, per-experiment data dirs), committed `.pyc`/egg-info
  removed.

Implication for Friday: if Friday runs the same hday ROS2 middleware, the Holly server is the
template — swap topics/services to Friday's namespaces, drop FR3-specific gripper action
clients for the hand's grasp primitives, and duplicate state subscriptions per arm.
