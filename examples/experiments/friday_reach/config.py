"""friday_reach — HIL-SERL on the friday bimanual arms (twin or real).

Quest grip = intervene, X = episode end. 12D action (2 arms x 6D EE delta),
no gripper/finger/waist action (dual-arm-fixed-gripper).

ALL shared knobs are at the top — edit once, applies to both arms. Per-arm
classes only carry what genuinely differs (server URL, reset pose, cameras).
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.friday_arm_env import FridayArmEnv, FridayArmConfig
from franka_env.envs.dual_franka_env import DualFridayEnv
from franka_env.envs.quest_intervention import QuestIntervention
from franka_env.envs.relative_env import DualRelativeFrame
from franka_env.envs.wrappers import (
    DualQuat2EulerWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig

# ---------------------------------------------------------------------------
# Shared knobs — ONE place, both arms.
# ---------------------------------------------------------------------------
# Teleop exaggeration: robot displacement converges to RATIO x hand
# displacement while gripped. 1.0 = 1:1 mimic.
TELEOP_EXAGGERATE_POS = 2.0
TELEOP_EXAGGERATE_ROT = 1.0


# Workspace box half-width around each arm's RESET_POSE (xyz m, euler rad).
POSE_LIMIT_DELTA = np.array([2.0, 2.0, 2.0, 1.5708, 1.5708, 1.5708])


class _ArmConfig(FridayArmConfig):
    """Everything both arms share."""

    # (xyz m, rot rad, unused) per unit action per step — the policy's action
    # space AND the teleop speed cap (max target motion = scale x hz).
    ACTION_SCALE = (1.0, 0.7854, 1)
    IMAGE_CROP = {}
    TARGET_POSE = np.zeros((6,))       # unused: reward comes from the classifier
    REWARD_THRESHOLD = np.zeros((6,))  # zeros => env reward never fires
    RANDOM_RESET = False
    # Episodes end ONLY on operator X press; restore ~120 for real RL runs.
    MAX_EPISODE_LENGTH = 100000
    DISPLAY_IMAGE = False
    JOINT_RESET_PERIOD = 0


class LeftArmConfig(_ArmConfig):
    SERVER_URL = "http://127.0.0.1:5010/left/"
    # Head stereo lives on the LEFT env only. Twin (alone_v1) has no stereo ->
    # empty. Real robot: {"stereo_left": {}, "stereo_right": {}} (required for
    # classifier/demo/RL, which need images).
    REALSENSE_CAMERAS = {}
    # MEASURED 2026-07-30, impedance twin resting pose (scipy euler xyz).
    # NOTE left yaw (-2.925) sits 0.217 rad from the -pi wrap.
    RESET_POSE = np.array([0.411, -0.390, 1.279, 2.158, 0.181, -2.925])
    ABS_POSE_LIMIT_LOW = RESET_POSE - POSE_LIMIT_DELTA
    ABS_POSE_LIMIT_HIGH = RESET_POSE + POSE_LIMIT_DELTA


class RightArmConfig(_ArmConfig):
    SERVER_URL = "http://127.0.0.1:5010/right/"
    REALSENSE_CAMERAS = {}
    RESET_POSE = np.array([-0.411, -0.389, 1.279, 0.981, 0.186, -0.215])
    ABS_POSE_LIMIT_LOW = RESET_POSE - POSE_LIMIT_DELTA
    ABS_POSE_LIMIT_HIGH = RESET_POSE + POSE_LIMIT_DELTA


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left/stereo_left", "left/stereo_right"]
    classifier_keys = ["left/stereo_left"]
    proprio_keys = ["left/tcp_pose", "right/tcp_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "dual-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        left_env = FridayArmEnv(
            fake_env=fake_env, save_video=save_video, config=LeftArmConfig,
        )
        right_env = FridayArmEnv(
            fake_env=fake_env, save_video=save_video, config=RightArmConfig,
        )
        env = DualFridayEnv(left_env, right_env, display_images=False)
        if not fake_env:
            env = QuestIntervention(
                env,
                pos_exaggerate=TELEOP_EXAGGERATE_POS,
                rot_exaggerate=TELEOP_EXAGGERATE_ROT,
            )
        env = DualRelativeFrame(env)
        env = DualQuat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier and self.classifier_keys is not None:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs)) > 0.75)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
