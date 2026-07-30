"""Meta Quest controller reader for teleop intervention.

Dependency-free (stdlib + numpy + scipy): listens for the Quest streaming app
(remy2 hand_tracking_streamer / remy3, controller mode) and exposes per-side
controller pose in robot-BASE-aligned axes plus button state.

IMPORTANT: this side is the SERVER. The Quest app is the client that connects
in and streams newline-delimited CSV. Bind before starting the app; keep
accepting so headset reconnects work.

Wire grammar (one line per controller per tick, plus a head line):

    Left controller:, x, y, z, qx, qy, qz, qw, trigger, grip, X, Y
    Right controller:, x, y, z, qx, qy, qz, qw, trigger, grip, A, B
    head:, x, y, z, qx, qy, qz, qw

Positions are meters in Unity world frame (Y-up, LEFT-handed); quaternions are
xyzw in the same frame. Buttons arrive as floats; pressed == value >= 0.5.

Frame handling: consumers of this reader use CLUTCH DELTAS (pose differences
from a snapshot), so the Unity world origin, the HMD calibration, and every
constant right-multiplied controller-to-wrist correction cancel exactly. The
only term that survives a world-frame delta is the constant LEFT rotation that
aligns Unity axes with the robot base:

    A = Rz(-base_yaw_deg) @ Rx(+90 deg)        (after the LH->RH mirror)

matching remy2's compute_wrist_pose step 3 (YUP_R_SIM @ Rz(yaw), inverted,
with base_yaw_deg = 180 for the friday mk3 profile). We apply A to BOTH p and
R here at the source, so downstream world-frame deltas are automatically
expressed in BASE axes (delta-R conjugates itself: A dR A^T).

LH->RH mirror (mirror Z): pos (x, y, z) -> (x, y, -z); quat (qx,qy,qz,qw) ->
(-qx, -qy, qz, qw). Do not touch qz — the negated pair is qx, qy.
"""

import logging
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
# Drop a side whose last controller line is older than this (headset asleep,
# app closed, tracking lost). Same don't-cry-wolf window remy2 uses.
DEFAULT_MAX_AGE_S = 0.25


def base_alignment(base_yaw_deg: float) -> np.ndarray:
    """Constant left rotation taking RH Y-up Unity vectors into robot BASE axes."""
    return (
        Rotation.from_euler("z", -base_yaw_deg, degrees=True)
        * Rotation.from_euler("x", 90, degrees=True)
    ).as_matrix()


def unity_to_base(pos_lh, quat_xyzw_lh, A: np.ndarray):
    """One Unity LH sample -> (p, R) in BASE-aligned axes.

    Applying A to both p and R makes world-frame deltas downstream come out in
    BASE axes with no further frame handling (dR_base = A dR_rh A^T).
    """
    p_rh = np.array([pos_lh[0], pos_lh[1], -pos_lh[2]], dtype=np.float64)
    q = quat_xyzw_lh
    R_rh = Rotation.from_quat([-q[0], -q[1], q[2], q[3]]).as_matrix()
    return A @ p_rh, A @ R_rh


@dataclass
class _ControllerState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))       # Unity LH
    quat_xyzw: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0, 1]))
    trigger: float = 0.0        # PrimaryIndexTrigger analog [0, 1]
    grip: float = 0.0           # PrimaryHandTrigger analog [0, 1]
    btn_primary: bool = False   # A (right) / X (left)
    btn_secondary: bool = False # B (right) / Y (left)
    last_update: float = 0.0    # host time.monotonic() at parse


class QuestReader:
    """Threaded Quest controller receiver.

    poses()   -> {side: (p (3,), R (3,3))} in BASE-aligned axes, fresh sides only.
    buttons() -> {side: {"trigger","grip","primary","secondary"}} (always both sides).
    """

    def __init__(self, protocol: str = "tcp", host: str = "0.0.0.0",
                 port: int = 8000, base_yaw_deg: float = 180.0,
                 max_age_s: float = DEFAULT_MAX_AGE_S):
        if protocol not in ("tcp", "udp"):
            raise ValueError(f"protocol must be tcp or udp, got {protocol!r}")
        self.protocol = protocol
        self.host = host
        self.port = port
        self._A = base_alignment(base_yaw_deg)
        self._max_age = float(max_age_s)
        self._lock = threading.Lock()
        self._state = {side: _ControllerState() for side in SIDES}
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle --------------------------------------------------------
    def start(self):
        run = self._run_tcp if self.protocol == "tcp" else self._run_udp
        self._thread = threading.Thread(target=run, name="quest:reader", daemon=True)
        self._thread.start()
        logger.info("QuestReader listening (%s) on %s:%d", self.protocol, self.host, self.port)

    def stop(self):
        self._stop.set()

    # -- consumer API -----------------------------------------------------
    def poses(self):
        """Fresh per-side controller pose in BASE-aligned axes."""
        now = time.monotonic()
        out = {}
        with self._lock:
            for side, st in self._state.items():
                if st.last_update and (now - st.last_update) <= self._max_age:
                    out[side] = unity_to_base(st.pos, st.quat_xyzw, self._A)
        return out

    def buttons(self):
        with self._lock:
            return {
                side: {
                    "trigger": st.trigger,
                    "grip": st.grip,
                    "primary": st.btn_primary,
                    "secondary": st.btn_secondary,
                }
                for side, st in self._state.items()
            }

    def connected(self) -> bool:
        now = time.monotonic()
        with self._lock:
            return any(
                st.last_update and (now - st.last_update) <= self._max_age
                for st in self._state.values()
            )

    # -- wire parsing -----------------------------------------------------
    def _handle_line(self, line: str):
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            return
        label = parts[0].lower()
        # Substring match on purpose: real labels are "Left controller:" etc.
        side = "right" if "right" in label else "left" if "left" in label else ""
        if not side or "controller" not in label:
            return  # head / body / landmark lines: not needed for clutch teleop
        floats = []
        for p in parts[1:]:
            if not p:
                continue
            try:
                floats.append(float(p))
            except ValueError:
                continue
        if len(floats) < 11:
            return
        v = np.asarray(floats, dtype=np.float64)
        with self._lock:
            st = self._state[side]
            st.pos = v[:3].copy()
            st.quat_xyzw = v[3:7].copy()
            st.trigger = float(v[7])
            st.grip = float(v[8])
            st.btn_primary = v[9] >= 0.5   # buttons are floats on the wire
            st.btn_secondary = v[10] >= 0.5
            st.last_update = time.monotonic()

    # -- transports (pattern copied from remy2 vr_app_receiver.py) --------
    def _run_tcp(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        server_sock.settimeout(0.5)
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = server_sock.accept()
                except socket.timeout:
                    continue
                logger.info("Quest app connected from %s", addr)
                t = threading.Thread(target=self._handle_tcp_conn, args=(conn,), daemon=True)
                t.start()
        finally:
            server_sock.close()

    def _handle_tcp_conn(self, conn: socket.socket):
        with conn:
            conn.settimeout(0.5)
            buf = ""
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break  # app disconnected; the accept loop takes the reconnect
                try:
                    buf += data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                # recv chunks are NOT line-aligned: keep the tail in the buffer.
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line:
                        self._handle_line(line)
        logger.info("Quest app disconnected")

    def _run_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.5)
        try:
            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                try:
                    message = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                # One datagram may carry several lines.
                for line in message.splitlines():
                    if line:
                        self._handle_line(line)
        finally:
            sock.close()


if __name__ == "__main__":
    # Smoke check: run, wave the controllers, watch poses stream.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reader = QuestReader()
    reader.start()
    try:
        while True:
            time.sleep(0.5)
            poses = reader.poses()
            btns = reader.buttons()
            for side, (p, _R) in poses.items():
                b = btns[side]
                print(f"{side}: p={np.round(p, 3)} grip={b['grip']:.2f} "
                      f"trig={b['trigger']:.2f} A/X={b['primary']} B/Y={b['secondary']}")
            if not poses:
                print("no fresh controller data (is the Quest app streaming?)")
    except KeyboardInterrupt:
        reader.stop()
