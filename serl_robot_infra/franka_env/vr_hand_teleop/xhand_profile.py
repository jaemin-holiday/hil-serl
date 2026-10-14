"""RobotEra XHand1 hand profile for remy2's retargeting, registered from here.

remy2 discovers hand profiles only from its own `retargeting/profiles/`
package, and vr_teleop resolves `--robot` only against its own
`configs/robots/`. This module injects the XHand profile into the registry at
runtime so remy2 stays an untouched, pinned dependency and every rby1-specific
file lives in this directory (see run_vr_teleop.py).

12 DoF, vendor joint names (side-prefixed in the URDF, so remy2's
`right_` → `left_` mirror applies as-is). mid/ring/pinky have no abduction
and only two flexion joints, so the landmark maps are built by hand: the 5x4
template helper would put 8 phantom joints on the wire that the IK never
overrides. The landmark baseline feeds only `--retargeter noop`; the IK
(xhand_right.yaml) overrides all 12 keys.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
IK_CONFIG = HERE / "xhand_right.yaml"
ROBOT_YAML = HERE / "rby1_xhand.robot.yaml"
ROUTING_YAML = HERE / "rby1_xhand.routing.yaml"
NAME = "rby1_xhand"


def build_profile():
    from retargeting.hand_profiles import (
        ABDUCTION_LANDMARK_IDS,
        FLEXION_LANDMARK_TRIPLETS,
        HandProfile,
    )
    from retargeting.robotics_math import normalize_angle

    abduction = {
        "right_hand_thumb_bend_joint": ABDUCTION_LANDMARK_IDS[1],
        "right_hand_index_bend_joint": ABDUCTION_LANDMARK_IDS[2],
    }
    flexion = {
        # thumb: rota1 <- MCP triplet, rota2 <- IP triplet
        "right_hand_thumb_rota_joint1": FLEXION_LANDMARK_TRIPLETS[1][1],
        "right_hand_thumb_rota_joint2": FLEXION_LANDMARK_TRIPLETS[1][2],
        # fingers: joint1 <- MCP flexion, joint2 <- PIP flexion
        "right_hand_index_joint1": FLEXION_LANDMARK_TRIPLETS[2][0],
        "right_hand_index_joint2": FLEXION_LANDMARK_TRIPLETS[2][1],
        "right_hand_mid_joint1": FLEXION_LANDMARK_TRIPLETS[3][0],
        "right_hand_mid_joint2": FLEXION_LANDMARK_TRIPLETS[3][1],
        "right_hand_ring_joint1": FLEXION_LANDMARK_TRIPLETS[4][0],
        "right_hand_ring_joint2": FLEXION_LANDMARK_TRIPLETS[4][1],
        "right_hand_pinky_joint1": FLEXION_LANDMARK_TRIPLETS[5][0],
        "right_hand_pinky_joint2": FLEXION_LANDMARK_TRIPLETS[5][1],
    }
    return HandProfile(
        name=NAME,
        abduction_mapping=abduction,
        flexion_landmarks=flexion,
        qpos_map_fn=lambda joint_name, angle: float(normalize_angle(angle)),
        # Bundled-name slot; the real config is passed as --ik-config IK_CONFIG.
        ik_config="xhand_right.yaml",
        # Left URDF already carries the mirrored axes (thumb bend 0 0 +-1, thumb
        # rota +-1 0 0, index bend -+1 0 0; flexion 0 1 0 both sides), so the
        # plain right_->left_ value mirror is correct: no qpos_sign.
    )


def register():
    """Put the profile into remy2's registry (idempotent). Call before
    load_robot_profile(ROBOT_YAML)."""
    import retargeting.hand_profiles as hp

    hp._ensure_loaded()
    if NAME not in hp.PROFILES:
        hp.PROFILES[NAME] = build_profile()
    return hp.PROFILES[NAME]


def load_robot_profile():
    from retargeting.robot_profiles import load_robot_profile as _load

    register()
    return _load(ROBOT_YAML)
