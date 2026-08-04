"""Robot server for the friday bimanual arms + head stereo camera.

Mirrors the franka_server.py pattern (one Flask app, one shared rclpy node
spun in a daemon thread) but serves BOTH arms from one process, with per-arm
routes under /left/... and /right/... — the gym env's SERVER_URL is then
    http://127.0.0.1:5000/left/     (LeftArmConfig)
    http://127.0.0.1:5000/right/    (RightArmConfig)
and the stock `url + "pose"` etc. calls compose correctly.

ROS interface (all names are flags — the state topics are placeholders until
the real EE-pose feedback topic exists):

  publish   geometry_msgs/PoseStamped  --cmd_topic_left / --cmd_topic_right
            (absolute EE pose target for the jacobian-transpose Cartesian
            impedance controller; quat xyzw in msg fields)
  subscribe geometry_msgs/PoseStamped  --state_topic_left / --state_topic_right
            (current EE pose per arm)
  subscribe sensor_msgs/CompressedImage --stereo_topic
            (head stereo, both eyes side by side in one JPEG)

Routes per arm (POST): pose {"arr": [x,y,z,qx,qy,qz,qw]}, getstate,
clearerr (no-op, kept because the env posts it every step), update_param
(no-op stub: impedance gains are owned by the controller side), jointreset
(no-op stub). Shared: GET /getimage -> latest stereo JPEG bytes.

getstate returns zeros for vel/force/torque/jacobian/q/dq/gripper_pos until
real feedback exists — the env only relies on "pose". It returns HTTP 503
until the first state message arrives, which the env turns into a clear
startup error instead of serving a stale zero pose.

Run (in the ROS environment, NOT the conda env):
    python friday_server.py
"""

import os
import threading

import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from absl import app, flags
from flask import Flask, jsonify, request, Response
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, JointState

FLAGS = flags.FLAGS
flags.DEFINE_string("flask_url", "127.0.0.1", "Flask host")
flags.DEFINE_integer("flask_port", 5000, "Flask port")
# Canonical names from message_names.h (kImpedanceTarget*Topic) — consumed by
# both the real driver and the digital twin, so no remap needed on either.
flags.DEFINE_string("cmd_topic_left", "/hday/friday/impedance_target_left",
                    "left arm EE pose target topic")
flags.DEFINE_string("cmd_topic_right", "/hday/friday/impedance_target_right",
                    "right arm EE pose target topic")
flags.DEFINE_string("state_topic_left", "/hday/friday/left_ee_pose",
                    "left arm EE pose state topic (PLACEHOLDER — set to the real one)")
flags.DEFINE_string("state_topic_right", "/hday/friday/right_ee_pose",
                    "right arm EE pose state topic (PLACEHOLDER — set to the real one)")
flags.DEFINE_string("stereo_topic", "/hday/sensor/camera/stereo_head/image/compressed",
                    "head stereo CompressedImage topic (gen2 mk1: stereo_head; "
                    "stereo_body is the separate body-mounted camera)")
flags.DEFINE_string("cmd_frame_id", "BASE", "frame_id stamped on pose commands")
# FK mode: compute the EE pose HERE from /joint_state + URDF instead of
# requiring the robot to publish EE-pose topics (state_topic_* are then
# ignored). Enable by passing --fk_urdf.
_BUNDLED_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "franka_env", "descriptions", "friday_gen1_mk5.urdf",
)
flags.DEFINE_string("fk_urdf", _BUNDLED_URDF,
                    "URDF for FK-based EE pose (default: the copy bundled in this "
                    "repo — kinematics only, no meshes needed). Pass an empty "
                    "string to fall back to the state_topic_* subscriptions.")
flags.DEFINE_string("fk_joint_state_topic", "/hday/friday/joint_state",
                    "JointState topic for FK mode")
flags.DEFINE_string("fk_root_link", "BASE", "FK base frame (must be the frame "
                    "the impedance controller interprets pose commands in)")
flags.DEFINE_string("fk_ee_link_left", "left_arm_ee", "left EE link name")
flags.DEFINE_string("fk_ee_link_right", "right_arm_ee", "right EE link name")


class UrdfChainFK:
    """Minimal FK for one fixed URDF chain (revolute + fixed joints).

    Parses the URDF once, walks child->parent from the EE link to the root
    link, and evaluates T = prod(T_origin(xyz, rpy) . Rot(axis, q)). URDF rpy
    is fixed-axis (extrinsic) xyz — scipy lowercase "xyz".
    """

    def __init__(self, urdf_path: str, root_link: str, ee_link: str):
        tree = ET.parse(urdf_path)
        by_child = {}
        for j in tree.getroot().iter("joint"):
            origin = j.find("origin")
            xyz = np.fromstring((origin.get("xyz", "0 0 0") if origin is not None
                                 else "0 0 0"), sep=" ")
            rpy = np.fromstring((origin.get("rpy", "0 0 0") if origin is not None
                                 else "0 0 0"), sep=" ")
            axis_el = j.find("axis")
            axis = np.fromstring(axis_el.get("xyz"), sep=" ") if axis_el is not None \
                else np.array([1.0, 0, 0])
            by_child[j.find("child").get("link")] = {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "T_origin": self._se3(Rotation.from_euler("xyz", rpy).as_matrix(), xyz),
                "axis": axis,
            }
        chain = []
        link = ee_link
        while link != root_link:
            if link not in by_child:
                raise ValueError(f"no path from {ee_link!r} to {root_link!r} "
                                 f"in {urdf_path} (stuck at {link!r})")
            joint = by_child[link]
            chain.append(joint)
            link = joint["parent"]
        self.chain = list(reversed(chain))
        self.joint_names = [j["name"] for j in self.chain
                            if j["type"] in ("revolute", "continuous")]

    @staticmethod
    def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def fk(self, q_by_name: dict) -> np.ndarray:
        """q_by_name: joint name -> angle. Returns pose [xyz + quat xyzw]
        of the EE link in the root-link frame."""
        T = np.eye(4)
        for j in self.chain:
            T = T @ j["T_origin"]
            if j["type"] in ("revolute", "continuous"):
                Tj = np.eye(4)
                Tj[:3, :3] = Rotation.from_rotvec(
                    j["axis"] * q_by_name[j["name"]]).as_matrix()
                T = T @ Tj
        return np.concatenate(
            [T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]
        )


class FridayArm:
    """One arm: pose command publisher + EE state subscription."""

    def __init__(self, node: Node, cmd_topic: str, state_topic: str = None):
        self.node = node
        self.pos = None  # [x,y,z,qx,qy,qz,qw]; None until first state msg
        self.pub = node.create_publisher(PoseStamped, cmd_topic, 10)
        # sensor-data QoS (BEST_EFFORT): the hday sensor stack publishes with
        # it (see remy2 model_inference ros2_source.py) — a default RELIABLE
        # subscription would match nothing and this server would 503 forever.
        # state_topic=None => FK mode: pos is written by the shared
        # joint-state callback in main() instead.
        if state_topic is not None:
            node.create_subscription(
                PoseStamped, state_topic, self._set_currpos, qos_profile_sensor_data
            )

    def _set_currpos(self, msg: PoseStamped):
        self.pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

    def move(self, pose):
        """Publish an absolute EE pose target: [x, y, z, qx, qy, qz, qw]."""
        assert len(pose) == 7
        msg = PoseStamped()
        msg.header.frame_id = FLAGS.cmd_frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = float(pose[2])
        msg.pose.orientation.x = float(pose[3])
        msg.pose.orientation.y = float(pose[4])
        msg.pose.orientation.z = float(pose[5])
        msg.pose.orientation.w = float(pose[6])
        self.pub.publish(msg)

    def state(self):
        if self.pos is None:
            return None
        # Everything but pose is a stub until real feedback topics exist.
        return {
            "pose": self.pos.tolist(),
            "vel": [0.0] * 6,
            "force": [0.0] * 3,
            "torque": [0.0] * 3,
            "q": [0.0] * 7,
            "dq": [0.0] * 7,
            "jacobian": np.zeros((6, 7)).tolist(),
            "gripper_pos": 0.0,
        }


def main(_):
    webapp = Flask(__name__)

    rclpy.init()
    node = Node("friday_server")

    if FLAGS.fk_urdf:
        # FK mode: no EE-pose topics needed from the robot — one JointState
        # subscription drives both arms' pose via URDF FK.
        arms = {
            "left": FridayArm(node, FLAGS.cmd_topic_left),
            "right": FridayArm(node, FLAGS.cmd_topic_right),
        }
        fk = {
            "left": UrdfChainFK(FLAGS.fk_urdf, FLAGS.fk_root_link, FLAGS.fk_ee_link_left),
            "right": UrdfChainFK(FLAGS.fk_urdf, FLAGS.fk_root_link, FLAGS.fk_ee_link_right),
        }
        needed = set(fk["left"].joint_names) | set(fk["right"].joint_names)
        missing_warned = [False]

        def _on_joint_state(msg: JointState):
            q = dict(zip(msg.name, msg.position))
            if not needed.issubset(q):
                if not missing_warned[0]:
                    print(f"FK: joint_state missing {sorted(needed - set(q))}; waiting")
                    missing_warned[0] = True
                return
            for side, chain in fk.items():
                arms[side].pos = chain.fk(q)

        node.create_subscription(
            JointState, FLAGS.fk_joint_state_topic, _on_joint_state,
            qos_profile_sensor_data,
        )
        print(f"friday_server: FK mode — {FLAGS.fk_joint_state_topic} + {FLAGS.fk_urdf}")
        print(f"friday_server: chains {FLAGS.fk_root_link} -> "
              f"{FLAGS.fk_ee_link_left} | {FLAGS.fk_ee_link_right}")
    else:
        arms = {
            "left": FridayArm(node, FLAGS.cmd_topic_left, FLAGS.state_topic_left),
            "right": FridayArm(node, FLAGS.cmd_topic_right, FLAGS.state_topic_right),
        }

    stereo = {"jpeg": None}

    def _set_stereo(msg: CompressedImage):
        stereo["jpeg"] = bytes(msg.data)

    node.create_subscription(
        CompressedImage, FLAGS.stereo_topic, _set_stereo, qos_profile_sensor_data
    )

    # -- per-arm routes ----------------------------------------------------
    @webapp.route("/<side>/pose", methods=["POST"])
    def pose(side):
        if side not in arms:
            return "unknown arm", 404
        arms[side].move(np.array(request.json["arr"]))
        return "Moved"

    @webapp.route("/<side>/getstate", methods=["POST"])
    def getstate(side):
        if side not in arms:
            return "unknown arm", 404
        state = arms[side].state()
        if state is None:
            return jsonify({"error": "no EE pose received yet"}), 503
        return jsonify(state)

    @webapp.route("/<side>/clearerr", methods=["POST"])
    def clearerr(side):
        return "ok"  # no error-recovery concept on this stack; env posts it every step

    @webapp.route("/<side>/update_param", methods=["POST"])
    def update_param(side):
        return "ok"  # gains owned by the impedance controller side; accepted and ignored

    @webapp.route("/<side>/jointreset", methods=["POST"])
    def jointreset(side):
        return "ok"  # no joint-space reset path yet; reset flows through pose targets

    # -- shared routes -----------------------------------------------------
    @webapp.route("/getimage", methods=["GET"])
    def getimage():
        # Serve under BOTH bare and per-arm prefixes so an arm env's
        # url + "getimage" works no matter which arm carries the cameras.
        if stereo["jpeg"] is None:
            return jsonify({"error": "no stereo frame received yet"}), 503
        return Response(stereo["jpeg"], mimetype="image/jpeg")

    @webapp.route("/<side>/getimage", methods=["GET"])
    def getimage_side(side):
        return getimage()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    print(f"friday_server: cmd -> {FLAGS.cmd_topic_left} | {FLAGS.cmd_topic_right}")
    print(f"friday_server: state <- {FLAGS.state_topic_left} | {FLAGS.state_topic_right}")
    webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port)


if __name__ == "__main__":
    app.run(main)
