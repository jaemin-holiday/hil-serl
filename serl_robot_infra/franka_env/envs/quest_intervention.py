"""Quest-controller intervention wrapper for the dual-arm friday env.

Drop-in replacement for DualSpacemouseIntervention with the same output
contract (`(action, replaced)` from action(), `info["intervene_action"]`,
`info["left"]` / `info["right"]`), so train_rlpd / record_demos /
record_success_fail run untouched.

Chain placement (matches object_handover): directly above DualFridayEnv,
INSIDE DualRelativeFrame. In this slot actions are BASE-frame 12D
(left 0:6, right 6:12); DualRelativeFrame inverse-transforms
info["intervene_action"] to the EE frame on the way up, so stored
interventions live in the policy's own action space.

Intervention trigger = GRIP button, per side (analog >= threshold). Grip held
on a side -> that arm's action comes from the human; released -> policy
resumes. The step-level `intervened` flag is the OR of the two sides — a step
where only one grip is held stores the policy's action for the OTHER arm as
part of the "demonstration".

Coupling: position-error (DROID-style position-position teleop). On grip
press, snapshot controller pose AND robot EE pose ("clutch"). While held:

    action_xyz = ((hand_p - hand_p0) - (ee_p - ee_p0)) / ACTION_SCALE[0]
    action_rot = rotvec( (hand_R hand_R0^T) (ee_R ee_R0^T)^T ) / ACTION_SCALE[1]

clipped to [-1, 1]. Free space: the arm tracks and the error stays small.
Contact: the arm is blocked, the hand pushes past the surface, the standing
error commands a steady force through the impedance controller — penetration
depth is the force knob, exactly like held SpaceMouse deflection. The clip
means max commanded step equals the policy's max, so interventions never
produce state-action pairs the policy cannot reproduce.

Release drops the anchors ("lift the mouse off the pad"): reposition the hand
anywhere, press again, fresh snapshot, robot stays put. Since the error is
zero by construction at the press instant, handover cannot jump.
"""

import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation

from franka_env.quest_reader import QuestReader

SIDES = ("left", "right")  # action layout: left = [0:6], right = [6:12]


def coupling_action(hand, hand0, ee, ee0, pos_scale, rot_scale,
                    pos_gain=1.0, rot_gain=1.0):
    """Position-error coupling for one arm -> normalized 6D action.

    hand/hand0/ee/ee0: (p (3,), R (3,3)) current and at-clutch poses, all in
    the same (BASE-aligned) world frame. Returns (a_xyz, a_rot), each clipped
    to [-1, 1]. Identically zero when hand == hand0 and ee == ee0.

    pos_gain/rot_gain: teleop exaggeration — the hand delta is multiplied
    before differencing, so the robot converges to gain x hand displacement
    (rotation: the hand's delta rotation angle is scaled about its axis).
    Amplifies operator motion WITHOUT touching ACTION_SCALE (the policy's
    action space) or the impedance gains (stability): the output stays a
    legal [-1, 1] action, only its saturation profile changes.
    """
    hand_p, hand_R = hand
    hand_p0, hand_R0 = hand0
    ee_p, ee_R = ee
    ee_p0, ee_R0 = ee0

    err_p = pos_gain * (hand_p - hand_p0) - (ee_p - ee_p0)
    a_xyz = np.clip(err_p / pos_scale, -1.0, 1.0)

    # World-frame left deltas; mismatch rotation -> axis-angle.
    dR_hand = Rotation.from_matrix(hand_R @ hand_R0.T)
    if rot_gain != 1.0:
        dR_hand = Rotation.from_rotvec(rot_gain * dR_hand.as_rotvec())
    dR_ee = ee_R @ ee_R0.T
    err_R = Rotation.from_matrix(dR_hand.as_matrix() @ dR_ee.T).as_rotvec()
    a_rot = np.clip(err_R / rot_scale, -1.0, 1.0)
    return a_xyz, a_rot


class QuestIntervention(gym.ActionWrapper):
    def __init__(self, env, reader: QuestReader = None, grip_threshold=0.5,
                 action_scale_left=None, action_scale_right=None,
                 pos_exaggerate=1.0, rot_exaggerate=1.0):
        super().__init__(env)
        assert self.action_space.shape == (12,), \
            "QuestIntervention expects the 12D dual-arm fixed-gripper action space"
        self.reader = reader or QuestReader()
        if reader is None:
            self.reader.start()
        self.grip_threshold = grip_threshold
        self.pos_exaggerate = float(pos_exaggerate)
        self.rot_exaggerate = float(rot_exaggerate)
        # Coupling gain = 1/ACTION_SCALE, so a hand-vs-robot mismatch of one
        # ACTION_SCALE saturates the action — the same per-step authority the
        # policy has. Defaults read from the wrapped arm envs.
        self._scales = {
            "left": np.asarray(
                action_scale_left
                if action_scale_left is not None
                else env.unwrapped.env_left.action_scale[:2], dtype=np.float64),
            "right": np.asarray(
                action_scale_right
                if action_scale_right is not None
                else env.unwrapped.env_right.action_scale[:2], dtype=np.float64),
        }
        self._engaged = {s: False for s in SIDES}
        self._hand0 = {}   # side -> (p, R) controller pose at grip press
        self._ee0 = {}     # side -> (p, R) robot EE pose at grip press
        self.left, self.right = False, False  # primary buttons, passed via info
        self._prev_x = False  # X rising-edge -> operator-triggered episode end

    def _ee_pose(self, side):
        arm = self.env.unwrapped.env_left if side == "left" else self.env.unwrapped.env_right
        pose = arm.currpos  # xyz + quat xyzw, base frame, refreshed each step by the arm env
        return np.asarray(pose[:3], dtype=np.float64), Rotation.from_quat(pose[3:]).as_matrix()

    def action(self, action: np.ndarray):
        """Returns (action, replaced). Grip-held sides get the coupling action."""
        poses = self.reader.poses()
        btns = self.reader.buttons()
        self.left = btns["left"]["primary"]     # X
        self.right = btns["right"]["primary"]   # A

        new_action = np.asarray(action, dtype=np.float64).copy()
        intervened = False

        for i, side in enumerate(SIDES):
            held = btns[side]["grip"] >= self.grip_threshold
            hand = poses.get(side)

            if not held or hand is None:
                # Released (or tracking lost): drop the anchors so the next
                # press re-latches wherever hand and robot then are.
                self._engaged[side] = False
                continue

            hand_p, hand_R = hand
            ee_p, ee_R = self._ee_pose(side)

            if not self._engaged[side]:
                self._hand0[side] = (hand_p.copy(), hand_R.copy())
                self._ee0[side] = (ee_p.copy(), ee_R.copy())
                self._engaged[side] = True
                # Error is identically zero at the press instant; fall through
                # and compute it anyway (no special case, no jump).

            pos_scale, rot_scale = self._scales[side]
            a_xyz, a_rot = coupling_action(
                (hand_p, hand_R), self._hand0[side],
                (ee_p, ee_R), self._ee0[side],
                pos_scale, rot_scale,
                pos_gain=self.pos_exaggerate, rot_gain=self.rot_exaggerate,
            )
            new_action[6 * i: 6 * i + 3] = a_xyz
            new_action[6 * i + 3: 6 * i + 6] = a_rot
            intervened = True

        if intervened:
            return new_action, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        # Operator-triggered episode end: X (left primary) rising edge.
        # `truncated`, not `done` — an operator abort is a timeout-style end,
        # so RL bootstrapping (masks = 1 - done) stays correct. A (right
        # primary) remains a pure label channel.
        if self.left and not self._prev_x:
            truncated = True
        self._prev_x = self.left
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self._engaged = {s: False for s in SIDES}
        self._hand0.clear()
        self._ee0.clear()
        return self.env.reset(**kwargs)

    def close(self):
        self.reader.stop()
        return self.env.close()
