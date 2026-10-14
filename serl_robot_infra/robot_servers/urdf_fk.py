"""Minimal URDF forward kinematics for one chain. Shared by the robot servers
(friday_server.py, rby1_server.py); no ROS imports so it is testable anywhere."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


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
