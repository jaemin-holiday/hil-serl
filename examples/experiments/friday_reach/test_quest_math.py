"""Self-checks for the Quest teleop math. Pure numpy/scipy — runs on any
python with those installed, no jax/gym/ros needed:

    python test_quest_math.py
"""

import sys
import types
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

# Make franka_env importable without installing, and stub gymnasium so
# quest_intervention (whose ActionWrapper we don't exercise) imports cleanly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "serl_robot_infra"))
for name, attrs in (
    ("gymnasium", {"ActionWrapper": type("ActionWrapper", (), {}),
                   "Env": type("Env", (), {}),
                   "spaces": types.SimpleNamespace(Box=object, Dict=object)}),
    ("pyrealsense2", {}),  # pulled in via franka_env.envs.__init__ -> rs_capture
):
    try:
        __import__(name)
    except ImportError:
        stub = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(stub, k, v)
        sys.modules[name] = stub

from franka_env.quest_reader import base_alignment, unity_to_base  # noqa: E402
from franka_env.envs.quest_intervention import coupling_action  # noqa: E402


def test_up_yaw_delta_maps_to_base_z_yaw():
    """A single pose A@R legitimately mixes axes (A is the frame remap); the
    invariant is on the world-frame DELTA: a yaw about Unity up (Y) between
    two samples must conjugate to a pure yaw about BASE up (+Z), same
    magnitude, proper rotation."""
    A = base_alignment(180.0)
    q1 = np.array([0.0, 0, 0, 1])                                  # identity
    q2 = Rotation.from_euler("y", -30, degrees=True).as_quat()      # 30 deg about up
    _, R1 = unity_to_base(np.zeros(3), q1, A)
    _, R2 = unity_to_base(np.zeros(3), q2, A)
    dR = R2 @ R1.T                                                  # = A dR_rh A^T
    assert np.isclose(np.linalg.det(dR), 1.0), "improper rotation — chirality broken"
    ang = Rotation.from_matrix(dR).as_rotvec()
    axis = ang / np.linalg.norm(ang)
    assert abs(abs(axis[2]) - 1.0) < 1e-9, f"up-yaw delta must be BASE z-yaw, axis={axis}"
    assert np.isclose(np.degrees(np.linalg.norm(ang)), 30.0), "angle magnitude changed"


def test_delta_frame_alignment():
    """Moving the hand forward in the room (Unity -Z at yaw 180 maps to BASE +X
    direction check) must produce a delta along a single BASE axis."""
    A = base_alignment(180.0)
    q = np.array([0.0, 0, 0, 1])
    p1, _ = unity_to_base(np.zeros(3), q, A)
    p2, _ = unity_to_base(np.array([0, 0, 0.5]), q, A)  # 0.5 m along Unity +Z
    d = p2 - p1
    # Unity +Z (LH forward) -> RH (0,0,-0.5) -> A @ : Rz(-180)Rx(90) maps
    # (0,0,-0.5) -> Rx90: (0, 0.5, 0) -> Rz(-180): (0, -0.5, 0).
    assert np.allclose(d, [0, -0.5, 0], atol=1e-12), f"forward delta wrong: {d}"


def test_right_multiplied_offsets_cancel_in_delta():
    """A constant controller-jig rotation offset (right-multiplied, like
    remy2's _CTRL_TO_WRIST_CORR) must cancel exactly in the world-frame
    rotation delta — the reason the reader needs no wrist correction."""
    A = base_alignment(180.0)
    rng = np.random.default_rng(0)
    q_offset = Rotation.random(random_state=rng)
    r1 = Rotation.random(random_state=rng)
    r2 = Rotation.random(random_state=rng)

    def to_base(r):
        _, R = unity_to_base(np.zeros(3), r.as_quat(), A)
        return R

    dR_plain = to_base(r2) @ to_base(r1).T
    dR_offset = to_base(r2 * q_offset) @ to_base(r1 * q_offset).T
    assert np.allclose(dR_plain, dR_offset, atol=1e-12), "right offset did not cancel"


def test_coupling_zero_at_latch():
    rng = np.random.default_rng(1)
    p = rng.normal(size=3)
    R = Rotation.random(random_state=rng).as_matrix()
    ep = rng.normal(size=3)
    eR = Rotation.random(random_state=rng).as_matrix()
    a_xyz, a_rot = coupling_action((p, R), (p, R), (ep, eR), (ep, eR), 0.02, 0.1)
    assert np.allclose(a_xyz, 0) and np.allclose(a_rot, 0), "nonzero action at latch"


def test_coupling_tracks_then_releases():
    """Hand moves 1 cm +x, robot hasn't -> positive x action. Robot catches
    up -> action returns to zero (self-healing, no lost motion)."""
    I = np.eye(3)
    hand0 = (np.zeros(3), I)
    ee0 = (np.array([0.5, 0, 0.3]), I)
    hand = (np.array([0.01, 0, 0]), I)

    a_xyz, _ = coupling_action(hand, hand0, ee0, ee0, 0.02, 0.1)
    assert np.allclose(a_xyz, [0.5, 0, 0]), f"expected half-scale +x, got {a_xyz}"

    ee_caught_up = (ee0[0] + np.array([0.01, 0, 0]), I)
    a_xyz, _ = coupling_action(hand, hand0, ee_caught_up, ee0, 0.02, 0.1)
    assert np.allclose(a_xyz, 0, atol=1e-12), "error must vanish when robot catches up"


def test_coupling_saturation_is_steady_force():
    """Hand held 3 cm past a blocked robot -> saturated, CONSTANT action —
    the held-deflection force analogy."""
    I = np.eye(3)
    hand0 = (np.zeros(3), I)
    ee0 = (np.zeros(3), I)
    hand = (np.array([0.03, 0, 0]), I)  # 3 cm past, scale 0.02 -> saturates
    a1, _ = coupling_action(hand, hand0, ee0, ee0, 0.02, 0.1)
    a2, _ = coupling_action(hand, hand0, ee0, ee0, 0.02, 0.1)
    assert np.allclose(a1, [1, 0, 0]) and np.allclose(a1, a2), "saturation not steady"


def test_coupling_exaggeration():
    """pos_gain scales the hand delta: 1 cm hand @ gain 2 == 2 cm error.
    rot_gain scales the hand delta ANGLE about its own axis."""
    I = np.eye(3)
    hand0 = (np.zeros(3), I)
    ee0 = (np.zeros(3), I)
    hand = (np.array([0.01, 0, 0]), I)
    a_xyz, _ = coupling_action(hand, hand0, ee0, ee0, 0.04, 0.1, pos_gain=2.0)
    assert np.allclose(a_xyz, [0.5, 0, 0]), f"2x1cm/0.04 should be 0.5, got {a_xyz}"

    Rh = Rotation.from_euler("x", 10, degrees=True).as_matrix()
    _, a_rot = coupling_action((np.zeros(3), Rh), (np.zeros(3), I),
                               ee0, ee0, 0.04, 0.5, rot_gain=2.0)
    expect = np.radians(20) / 0.5
    assert np.isclose(a_rot[0], expect), f"2x10deg/0.5rad: want {expect}, got {a_rot[0]}"


def test_coupling_rotation_direction():
    """Hand rolls +20 deg about world x, robot hasn't -> positive x rot action
    of the right magnitude (20 deg / 0.1 rad scale, clipped to 1)."""
    I = np.eye(3)
    Rh = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    _, a_rot = coupling_action((np.zeros(3), Rh), (np.zeros(3), I),
                               (np.zeros(3), I), (np.zeros(3), I), 0.02, 0.1)
    assert a_rot[0] == 1.0 and abs(a_rot[1]) < 1e-9 and abs(a_rot[2]) < 1e-9, \
        f"rot coupling wrong: {a_rot} (0.349 rad / 0.1 clips to 1)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
