"""XHand1 joint order and limits shared by the env side (ZmqQuestReader,
XHandPassthrough). The robot server (robot_servers/rby1_server.py) carries the
same table; it runs in another environment and cannot import this package."""

import numpy as np

# RobotEra XHand1 SDK / rby1_driver order (= holiday spec `<side>_hand` group).
HAND_JOINTS = (
    "thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2",
    "index_bend_joint", "index_joint1", "index_joint2",
    "mid_joint1", "mid_joint2", "ring_joint1", "ring_joint2",
    "pinky_joint1", "pinky_joint2",
)
HAND_LO = np.array([0.0, -0.698, 0.0, -0.174, 0, 0, 0, 0, 0, 0, 0, 0])
HAND_HI = np.array([1.832, 1.57, 1.57, 0.174] + [1.919] * 8)


def joint_name(side: str, short: str) -> str:
    return f"{side}_hand_{short}"


def vector_from_joint_angles(side: str, joint_angles: dict):
    """vr_teleop `hand.joint_angles` dict -> (12,) in SDK order, or None if any
    joint is missing (a hand profile other than rby1_xhand, or a locked hand)."""
    try:
        q = np.array([joint_angles[joint_name(side, n)] for n in HAND_JOINTS], dtype=np.float64)
    except (KeyError, TypeError):
        return None
    return np.clip(q, HAND_LO, HAND_HI)
