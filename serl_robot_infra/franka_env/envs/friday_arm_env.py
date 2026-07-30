"""Single-arm gym env for one friday arm, no gripper.

Subclass of the base FridayEnv (franka_env.py) with three deviations:

  * 6D action space — friday's fingers/waist are NOT part of the RL action
    (decision 2026-07-29); they hold a constant target owned by the robot
    side. No gripper HTTP calls, no gripper action dim. Pairs with
    setup_mode="dual-arm-fixed-gripper".
  * Cameras come from the friday robot server over HTTP (/getimage: latest
    head-stereo JPEG, both eyes side by side), not from pyrealsense. This
    keeps ALL ROS in the server process — this env runs in the conda/jax env.
  * fake_env=True never touches the network (the base class fetches state
    before its fake-env early return, forcing the learner to need a live
    robot server; here the learner runs standalone).

Talks to robot_servers/friday_server.py, whose per-arm routes live under
SERVER_URL = http://host:5000/<left|right>/ — the base class's `url + route`
calls compose the arm prefix for free.

Obs keys and shapes are kept identical to the base env (tcp_vel / force /
torque / gripper_pose are zero stubs until real feedback topics exist) so the
stock Dual* wrappers and SERLObsWrapper apply unchanged; select real signals
via proprio_keys.
"""

import time

import cv2
import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation

from franka_env.envs.franka_env import FridayEnv, DefaultEnvConfig


class FridayArmConfig(DefaultEnvConfig):
    """Per-arm config. REALSENSE_CAMERAS keys name views served by the robot
    server's stereo topic: "stereo_left" / "stereo_right" (values unused).
    Give the cameras to ONE arm env only — the dual env would duplicate them."""

    STEREO_TIMEOUT_S = 2.0


class FridayArmEnv(FridayEnv):
    def __init__(self, hz=10, fake_env=False, save_video=False,
                 config: FridayArmConfig = None, set_load=False):
        # Deliberately NOT calling super().__init__: it fetches robot state
        # before the fake-env early return and hardcodes the 7D gripper action
        # space. Attribute names below match the base class exactly so every
        # inherited method (clip_safety_box, go_to_reset, interpolate_move,
        # reset, _send_pos_command, _get_obs, compute_reward) works unchanged.
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = False  # display is owned by DualFridayEnv
        self.gripper_sleep = config.GRIPPER_SLEEP

        # scipy "xyz" euler -> quat xyzw, NOT the base env's euler_2_quat:
        # that helper returns wxyz (pyquaternion) with a pi-yaw offset and does
        # not invert Rotation.as_euler("xyz") — RESET_POSE values here are
        # measured via scipy, so they must be converted via scipy.
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3],
             Rotation.from_euler("xyz", config.RESET_POSE[3:]).as_quat()]
        )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD
        self.save_video = save_video
        if self.save_video:
            self.recording_frames = []

        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3], config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:], config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # 6D: xyz delta + euler delta. No gripper dim.
        self.action_space = gym.spaces.Box(
            np.ones((6,), dtype=np.float32) * -1,
            np.ones((6,), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                     for key in config.REALSENSE_CAMERAS}
                ),
            }
        )
        self.cycle_count = 0
        self.terminate = False

        if fake_env:
            # Learner-side spaces-only env: stub state, no network, no cameras.
            self.currpos = self.resetpos.copy()
            self.cmdpos = self.resetpos.copy()
            self.currvel = np.zeros(6)
            self.currforce = np.zeros(3)
            self.currtorque = np.zeros(3)
            self.currjacobian = np.zeros((6, 7))
            self.q = np.zeros(7)
            self.dq = np.zeros(7)
            self.curr_gripper_pos = np.zeros(1)
            return

        self._update_currpos()
        # Internal SE(3) command state. Seeded from the measured pose ONCE at
        # startup; from then on actions integrate it and it is never re-anchored
        # to measurement — commanding target := measured couples the controller
        # to its own noise through the loop delay (wrist shiver).
        self.cmdpos = self.currpos.copy()

        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        print(f"Initialized FridayArmEnv @ {self.url}")

    # -- action ------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple:
        """Base step minus the gripper: 6D normalized delta -> pose command."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Integrate the action on the INTERNAL command state, not the measured
        # pose (base-env behavior) — zero action = identical command every
        # tick, no measurement feedback.
        self.nextpos = self.cmdpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + action[:3] * self.action_scale[0]
        # World-frame delta rotation, pre-multiplied (matches the base env).
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.cmdpos[3:])
        ).as_quat()

        self.nextpos = self.clip_safety_box(self.nextpos)
        self._send_pos_command(self.nextpos)
        self.cmdpos = self.nextpos.copy()

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = (
            self.curr_path_length >= self.max_episode_length
            or reward
            or self.terminate
        )
        return ob, int(reward), done, False, {"succeed": reward}

    def _send_gripper_command(self, pos, mode="binary"):
        pass  # no gripper on this robot — fingers hold a constant target

    def go_to_reset(self, joint_reset=False):
        """Base version minus its `_send_pos_command(self.currpos)` opener —
        never command the measured pose (see cmdpos). Also uses scipy for the
        randomized euler and RESET_PARAM-free flow identical to base."""
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)
        if joint_reset:
            requests.post(self.url + "jointreset")
            time.sleep(0.5)
        reset_pose = self.resetpos.copy()
        if self.randomreset:
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = Rotation.from_euler("xyz", euler_random).as_quat()
        self.interpolate_move(reset_pose, timeout=1)
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Base version with the euler->quat conversion done via scipy
        (see resetpos comment); otherwise a faithful copy."""
        if goal.shape == (6,):
            goal = np.concatenate(
                [goal[:3], Rotation.from_euler("xyz", goal[3:]).as_quat()]
            )
        steps = int(timeout * self.hz)
        # Interpolate from the internal command state, land the command there.
        path = np.linspace(self.cmdpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self.cmdpos = np.asarray(goal, dtype=np.float64).copy()
        self._update_currpos()

    # -- state -------------------------------------------------------------
    def _update_currpos(self):
        resp = requests.post(self.url + "getstate")
        if resp.status_code != 200:
            raise RuntimeError(
                f"friday_server at {self.url} has no robot state yet "
                f"(HTTP {resp.status_code}). Is the EE-pose topic publishing?"
            )
        ps = resp.json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])
        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))
        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])
        self.curr_gripper_pos = np.array([ps["gripper_pos"]])

    # keep the public duplicate the base class exposes
    def update_currpos(self):
        self._update_currpos()

    # -- cameras (head stereo over HTTP, no pyrealsense) --------------------
    def init_cameras(self, name_serial_dict=None):
        pass  # nothing to open; frames come from the robot server per get_im

    def get_im(self):
        keys = list(self.observation_space["images"].spaces.keys())
        if not keys:
            return {}
        resp = requests.get(
            self.url + "getimage", timeout=self.config.STEREO_TIMEOUT_S
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"friday_server at {self.url} has no stereo frame yet "
                f"(HTTP {resp.status_code}). Is the camera topic publishing?"
            )
        frame = cv2.imdecode(
            np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        half = frame.shape[1] // 2
        eyes = {"stereo_left": frame[:, :half], "stereo_right": frame[:, half:]}

        images = {}
        full_res_images = {}
        for key in keys:
            raw = eyes[key]
            cropped = (
                self.config.IMAGE_CROP[key](raw)
                if key in self.config.IMAGE_CROP
                else raw
            )
            target = self.observation_space["images"][key].shape[:2][::-1]
            resized = cv2.resize(cropped, target)
            images[key] = resized[..., ::-1]  # BGR -> RGB, as the base env does
            full_res_images[key] = cropped
        if self.save_video:
            # One dict per step, keyed by camera — the shape the inherited
            # save_video_recording iterates over.
            self.recording_frames.append(full_res_images)
        return images

    def close(self):
        # No camera handles to release; do stop the ESC listener.
        if hasattr(self, "listener"):
            self.listener.stop()
