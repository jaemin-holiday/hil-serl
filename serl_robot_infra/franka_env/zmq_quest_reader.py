"""Hand-tracking input for teleop intervention: the remy2 vr_teleop ZMQ wire.

Same consumer API as quest_reader.QuestReader (controller mode), so
QuestIntervention takes either:

    poses()   -> {side: (p (3,), R (3,3))}   fresh sides whose wrist is UNLOCKED
    buttons() -> {side: {"trigger","grip","primary","secondary"}}
    fingers() -> {side: (12,) rad, XHand SDK order}   fresh sides with finger data

Mapping of the wire (vr_teleop `--robot rby1_xhand`, see franka_env/vr_hand_teleop):
  - `msg[side]["arm"]` present  -> grip = 1.0 (the operator typed wr/wl in the
    bridge; the wrist is live). Absent (locked / untracked) -> grip = 0.0, which
    QuestIntervention treats as release: anchors dropped, arm frozen. The bridge's
    wrist lock IS the deadman in hand mode.
  - `arm.position` / `arm.rotation` (wxyz): wrist pose already in the robot base
    frame with the profile's EE correction right-multiplied — QuestIntervention
    uses clutch deltas, so that constant cancels.
  - `hand.joint_angles`: 12 XHand joints per side -> fingers().
  - primary buttons do not exist on hand tracking: keyboard `x` = left primary
    (episode end, as the Quest X), `a` = right primary (label). pynput when a
    display is available; without it the keys are simply inactive.
"""

import logging
import pickle
import threading
import time

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from franka_env.xhand import vector_from_joint_angles

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
DEFAULT_MAX_AGE_S = 0.25   # vr_teleop publishes at the Quest rate (~60 Hz)
# vr_teleop keeps republishing its last snapshot after the headset stops
# sending (no staleness check on its side), so message age alone cannot tell a
# dead headset from a live one. Real tracking never repeats a wrist position
# bit for bit; a position unchanged this long is treated as lost.
DEFAULT_MAX_STILL_S = 0.5
KEY_PULSE_S = 0.3          # a key press reads as "pressed" this long (>= 2 env steps at 10 Hz)


def _f32(buf):
    return np.frombuffer(buf, dtype=np.float32).astype(np.float64)


class ZmqQuestReader:
    def __init__(self, address: str = "tcp://127.0.0.1:5555", max_age_s: float = DEFAULT_MAX_AGE_S,
                 max_still_s: float = DEFAULT_MAX_STILL_S):
        self.address = address
        self._max_age = float(max_age_s)
        self._max_still = float(max_still_s)
        self._lock = threading.Lock()
        self._arm = {s: None for s in SIDES}       # (p, R, t_recv, t_changed) or None
        self._last_p = {s: None for s in SIDES}
        self._changed = {s: 0.0 for s in SIDES}
        self._fingers = {s: None for s in SIDES}   # (q12, t) or None
        self._last_msg = 0.0
        self._key_until = {"left": 0.0, "right": 0.0}
        self._stop = threading.Event()
        self._thread = None
        self._kb = None

    # -- lifecycle --------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="quest:zmq", daemon=True)
        self._thread.start()
        self._start_keys()
        logger.info("ZmqQuestReader subscribed to %s (x = episode end, a = label)", self.address)

    def stop(self):
        self._stop.set()
        if self._kb is not None:
            try:
                self._kb.stop()
            except Exception:  # noqa: BLE001
                pass

    def _start_keys(self):
        try:
            from pynput import keyboard
        except Exception as e:  # noqa: BLE001 - no display / no pynput
            print(f"ZmqQuestReader: key listener disabled ({type(e).__name__}); no episode-end key")
            return

        def on_press(key):
            ch = getattr(key, "char", None)
            side = {"x": "left", "a": "right"}.get(ch)
            if side:
                with self._lock:
                    self._key_until[side] = time.monotonic() + KEY_PULSE_S

        self._kb = keyboard.Listener(on_press=on_press)
        self._kb.start()

    # -- wire -------------------------------------------------------------
    def _run(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.RCVTIMEO, 200)
        sub.connect(self.address)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        while not self._stop.is_set():
            try:
                raw = sub.recv()
            except zmq.Again:
                continue
            try:
                msg = pickle.loads(raw)
            except Exception as e:  # noqa: BLE001
                logger.warning("ZmqQuestReader: undecodable frame: %s", e)
                continue
            now = time.monotonic()
            with self._lock:
                self._last_msg = now
                for side in SIDES:
                    s = msg.get(side) or {}
                    arm = s.get("arm")
                    if arm is not None:
                        q = _f32(arm["rotation"])  # wxyz
                        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                        p = _f32(arm["position"])
                        if self._last_p[side] is None or not np.array_equal(p, self._last_p[side]):
                            self._changed[side] = now
                            self._last_p[side] = p
                        self._arm[side] = (p, R, now, self._changed[side])
                    else:
                        self._arm[side] = None
                        self._last_p[side] = None
                    ja = (s.get("hand") or {}).get("joint_angles")
                    vec = vector_from_joint_angles(side, ja) if ja else None
                    self._fingers[side] = (vec, now) if vec is not None else None
        sub.close(0)
        ctx.term()

    # -- consumer API -----------------------------------------------------
    def _fresh(self, t):
        return (time.monotonic() - t) <= self._max_age

    def _live(self, a):
        """Arm entry is recent AND still moving (see DEFAULT_MAX_STILL_S)."""
        return a is not None and self._fresh(a[2]) and (time.monotonic() - a[3]) <= self._max_still

    def poses(self):
        out = {}
        with self._lock:
            for side, a in self._arm.items():
                if self._live(a):
                    out[side] = (a[0].copy(), a[1].copy())
        return out

    def buttons(self):
        now = time.monotonic()
        with self._lock:
            return {
                side: {
                    "trigger": 0.0,
                    "grip": 1.0 if self._live(self._arm[side]) else 0.0,
                    "primary": now < self._key_until[side],
                    "secondary": False,
                }
                for side in SIDES
            }

    def fingers(self):
        out = {}
        with self._lock:
            for side, f in self._fingers.items():
                if f is not None and self._fresh(f[1]):
                    out[side] = f[0].copy()
        return out

    def connected(self) -> bool:
        with self._lock:
            return self._fresh(self._last_msg) if self._last_msg else False
