"""Robot server for RB-Y1 + XHand on the holiday controller (fabrics).

Same HTTP contract as friday_server.py, so the gym env is unchanged:
    http://127.0.0.1:5010/left/    http://127.0.0.1:5010/right/
Routes per arm (POST): pose {"arr": [x,y,z,qx,qy,qz,qw]}, getstate, clearerr,
update_param (no-ops), jointreset (arm back to the spec's `ready` state),
hand {"q": [12 rad, XHand SDK order]} (finger targets; getstate echoes them
back as "hand_q" from joint_states). Shared: GET /getimage (503 until a camera
topic exists on this stack).

ROS side = the holiday Manipulation API, the same wire vr_teleop and the
inference stack use on every holiday robot:

  publish   friday_msgs/Manipulation  holiday/manipulation   (stamped; LinkReaching
            for the arm = absolute pose of link_<side>_arm_6 in `base`, driven by
            <side>_arm_0..6 only; JointReaching for jointreset and for the fingers —
            finger joints are non-fabric "held" actuators in the controller, which
            passes them straight through to the driver's /hday_ctrl_cmd and on to
            the XHand over RS485; SIL has no hands, so there they are a no-op)
  subscribe sensor_msgs/JointState    holiday/joint_states   (BEST_EFFORT) ->
            URDF FK (base -> link_<side>_arm_6) = the EE pose getstate returns

EE frame = link_<side>_arm_6 (the wrist flange the hand bracket bolts to), NOT
<side>_hand_link: the controller spec's body_reaching slots are named after that
body (specs/robot/rby1_xhand/common.controller.yaml) and a LinkReaching whose
target_frame matches no slot is dropped without a log line. The hand base sits a
fixed 0.1363 m further along the flange -Z (bracket + mount), so a target on the
flange is a target on the hand up to that constant.

The controller must be rolling out (holiday/control/set_enabled true) for a
target to execute; this server never flips that switch.

Run inside the rby1-driver image (ROS jazzy + friday_msgs), see
scripts/rby1_server_docker.sh. Works against src/drivers/rby1-driver/sil.sh
(vendor simulator) and the real robot alike — only ROS_DOMAIN_ID/DDS differ.
"""

import math
import os
import threading

import numpy as np
import rclpy
from absl import app, flags
from flask import Flask, jsonify, request
from friday_msgs.msg import JointReaching, LinkReaching, Manipulation
from geometry_msgs.msg import Point, Quaternion
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from urdf_fk import UrdfChainFK  # sibling module; servers run as scripts

FLAGS = flags.FLAGS
flags.DEFINE_string("flask_url", "127.0.0.1", "Flask host")
flags.DEFINE_integer("flask_port", 5010, "Flask port")
flags.DEFINE_string("manipulation_topic", "holiday/manipulation", "Manipulation command topic")
flags.DEFINE_string("joint_state_topic", "holiday/joint_states", "JointState topic from the driver")
flags.DEFINE_string("reference_frame", "base", "LinkReaching reference frame (RobotSpec root body)")
flags.DEFINE_string("urdf", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "franka_env", "descriptions", "rby1_xhand.urdf"), "kinematics-only URDF for FK")
flags.DEFINE_float("position_weight", 5.0, "LinkReaching position_weight")
flags.DEFINE_float("orientation_weight", 1.0, "LinkReaching orientation_weight")

SIDES = ("left", "right")
ARM_JOINTS = {s: [f"{s}_arm_{i}" for i in range(7)] for s in SIDES}
EE_LINK = {s: f"link_{s}_arm_6" for s in SIDES}  # = the controller's body_reaching slot names
# XHand1 joints in the SDK/driver order (rby1_driver kXHandJointNames, = spec
# common.joint_groups.yaml `<side>_hand`), with the URDF limits.
HAND_JOINTS = ("thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2",
               "index_bend_joint", "index_joint1", "index_joint2",
               "mid_joint1", "mid_joint2", "ring_joint1", "ring_joint2",
               "pinky_joint1", "pinky_joint2")
HAND_LO = np.array([0.0, -0.698, 0.0, -0.174, 0, 0, 0, 0, 0, 0, 0, 0])
HAND_HI = np.array([1.832, 1.57, 1.57, 0.174] + [1.919] * 8)
# `ready` from specs/robot/rby1_xhand/common.named_states.yaml (Rainbow READY_POSE), arm part.
READY_Q = {
    "left": [0.0, 0.0872664626, 0.0, -2.0943951024, 0.0, 1.2217304764, 0.0],
    "right": [0.0, -0.0872664626, 0.0, -2.0943951024, 0.0, 1.2217304764, 0.0],
}


class Rby1Arm:
    """One arm: Manipulation publisher + FK-derived EE state."""

    def __init__(self, node: Node, side: str, pub):
        self.node, self.side, self.pub = node, side, pub
        self.fk = UrdfChainFK(FLAGS.urdf, FLAGS.reference_frame, EE_LINK[side])
        self.pos = None          # [x,y,z,qx,qy,qz,qw] in reference_frame
        self.q = [0.0] * 7
        self.dq = [0.0] * 7
        self.hand_q = [0.0] * 12
        self.hand_target = None   # last /hand target; re-sent with every /pose

    def _msg(self):
        m = Manipulation()
        m.header.stamp = self.node.get_clock().now().to_msg()
        return m

    def _hand_reaching(self, q):
        return [JointReaching(target_joint=f"{self.side}_hand_{name}", target_position=float(v))
                for name, v in zip(HAND_JOINTS, q)]

    def move(self, pose):
        """Absolute EE pose target [x, y, z, qx, qy, qz, qw] in reference_frame.

        The latest finger target rides along in the same message: the
        controller takes the newest Manipulation per tick, so a finger-only
        message sent just before a pose message could be skipped. One message
        per side per step carries both."""
        assert len(pose) == 7
        lr = LinkReaching()
        lr.target_frame = EE_LINK[self.side]
        lr.reference_frame = FLAGS.reference_frame
        lr.active_joints = ARM_JOINTS[self.side]
        lr.position_weight = float(FLAGS.position_weight)
        lr.orientation_weight = float(FLAGS.orientation_weight)
        lr.target_pos = Point(x=float(pose[0]), y=float(pose[1]), z=float(pose[2]))
        lr.target_quat = Quaternion(x=float(pose[3]), y=float(pose[4]), z=float(pose[5]), w=float(pose[6]))
        m = self._msg()
        m.link_reaching.append(lr)
        if self.hand_target is not None:
            m.joint_reaching.extend(self._hand_reaching(self.hand_target))
        self.pub.publish(m)

    def joint_reset(self):
        m = self._msg()
        for name, q in zip(ARM_JOINTS[self.side], READY_Q[self.side]):
            m.joint_reaching.append(JointReaching(target_joint=name, target_position=float(q)))
        self.pub.publish(m)

    def move_hand(self, q12):
        """12 finger targets (rad, SDK order), clipped to the URDF limits."""
        q = np.clip(np.asarray(q12, dtype=float), HAND_LO, HAND_HI)
        self.hand_target = q
        m = self._msg()
        m.joint_reaching.extend(self._hand_reaching(q))
        self.pub.publish(m)

    def update(self, q_by_name, dq_by_name):
        self.pos = self.fk.fk(q_by_name)
        self.q = [float(q_by_name[j]) for j in ARM_JOINTS[self.side]]
        self.dq = [float(dq_by_name.get(j, 0.0)) for j in ARM_JOINTS[self.side]]
        self.hand_q = [float(q_by_name.get(f"{self.side}_hand_{n}", 0.0)) for n in HAND_JOINTS]

    def state(self):
        if self.pos is None:
            return None
        return {
            "pose": self.pos.tolist(),
            "vel": [0.0] * 6,
            "force": [0.0] * 3,
            "torque": [0.0] * 3,
            "q": self.q,
            "dq": self.dq,
            "jacobian": np.zeros((6, 7)).tolist(),
            "gripper_pos": 0.0,
            "hand_q": self.hand_q,
        }


def main(_):
    webapp = Flask(__name__)
    rclpy.init()
    node = Node("rby1_server")
    pub = node.create_publisher(Manipulation, FLAGS.manipulation_topic, 1)
    arms = {s: Rby1Arm(node, s, pub) for s in SIDES}
    needed = {j for s in SIDES for j in arms[s].fk.joint_names}
    warned = [False]

    def _on_joint_state(msg: JointState):
        q = dict(zip(msg.name, msg.position))
        dq = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        if not needed.issubset(q):
            if not warned[0]:
                print(f"rby1_server: joint_states missing {sorted(needed - set(q))}; waiting")
                warned[0] = True
            return
        for arm in arms.values():
            arm.update(q, dq)

    node.create_subscription(
        JointState, FLAGS.joint_state_topic, _on_joint_state,
        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
    )

    @webapp.route("/<side>/pose", methods=["POST"])
    def pose(side):
        if side not in arms:
            return "unknown arm", 404
        arr = np.asarray(request.json["arr"], dtype=float)
        if arr.shape != (7,) or not np.all(np.isfinite(arr)):
            return "pose must be 7 finite floats [x,y,z,qx,qy,qz,qw]", 400
        arms[side].move(arr)
        return "Moved"

    @webapp.route("/<side>/getstate", methods=["POST"])
    def getstate(side):
        if side not in arms:
            return "unknown arm", 404
        state = arms[side].state()
        if state is None:
            return jsonify({"error": "no joint_states received yet"}), 503
        return jsonify(state)

    @webapp.route("/<side>/hand", methods=["POST"])
    def hand(side):
        if side not in arms:
            return "unknown arm", 404
        q = np.asarray(request.json["q"], dtype=float)
        if q.shape != (12,) or not np.all(np.isfinite(q)):
            return "q must be 12 finite floats (XHand SDK order)", 400
        arms[side].move_hand(q)
        return "ok"

    @webapp.route("/<side>/clearerr", methods=["POST"])
    def clearerr(side):
        return "ok"

    @webapp.route("/<side>/update_param", methods=["POST"])
    def update_param(side):
        return "ok"  # gains live in the controller spec

    @webapp.route("/<side>/jointreset", methods=["POST"])
    def jointreset(side):
        if side not in arms:
            return "unknown arm", 404
        arms[side].joint_reset()
        return "ok"

    @webapp.route("/getimage", methods=["GET"])
    @webapp.route("/<side>/getimage", methods=["GET"])
    def getimage(side=None):
        return jsonify({"error": "no camera on this stack yet"}), 503

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(f"rby1_server: cmd -> {FLAGS.manipulation_topic} (LinkReaching {EE_LINK} in {FLAGS.reference_frame!r})")
    print(f"rby1_server: state <- {FLAGS.joint_state_topic} + FK {FLAGS.urdf}")
    webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port)


if __name__ == "__main__":
    app.run(main)
