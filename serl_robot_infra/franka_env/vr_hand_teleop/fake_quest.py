#!/usr/bin/env python3
"""Fake Quest hand-tracking client: streams the Remy2 app's wire into vr_teleop.

For testing the whole hand-mode chain without a headset:
    fake_quest.py -> vr_teleop (run_vr_teleop.sh) -> ZMQ -> ZmqQuestReader -> env -> rby1_server -> controller

Both hands + head, 60 Hz, TCP to vr_teleop's :8000 (the PC is the server, so
start vr_teleop first). Motion: the wrists bob up and down (`--amp`, `--period`)
and the fingers open/close with the same period (`--curl`). Packet format =
hand_tracking_streamer/README.md "Packet Format": per hand a wrist line
(Unity world pos + quat xyzw), a landmarks line (21 x xyz, wrist-local) and a
rotations line (21 x quat); then `head:`.

Landmarks are authored in the retargeter's wrist-local Isaac frame (+Z distal,
+Y thumb, +X palm — the fixture in remy2 retargeting/tests) and mapped back to
Unity wrist-local with the inverse of SkeletonProcessor's (x, y, z) ->
(-y, -x, z), so the IK sees a plausible hand. Rotations are identity (only the
noop retargeter reads them).

    python fake_quest.py                 # until Ctrl+C
    python fake_quest.py --seconds 20 --amp 0.05 --period 6
"""

import argparse
import math
import socket
import threading
import time

import numpy as np

# Open-hand keypoints in the Isaac wrist frame (remy2 retargeting test fixture):
# index 3/7/11/15/19 = distal phalanx ("aux"), 4/8/12/16/20 = tip. Others are
# filled along the finger so the 21-joint array is complete.
_OPEN = {
    "thumb":  (np.array([0.05, 0.08, 0.08]), np.array([0.07, 0.10, 0.10])),
    "index":  (np.array([0.01, 0.05, 0.18]), np.array([0.01, 0.05, 0.21])),
    "middle": (np.array([0.01, 0.02, 0.19]), np.array([0.01, 0.02, 0.22])),
    "ring":   (np.array([0.01, -0.01, 0.18]), np.array([0.01, -0.01, 0.21])),
    "pinky":  (np.array([0.01, -0.03, 0.17]), np.array([0.01, -0.03, 0.20])),
}
# Closed hand: tips curl toward the palm (+X) and back toward the wrist (-Z).
_CLOSED = {
    "thumb":  (np.array([0.06, 0.05, 0.11]), np.array([0.07, 0.03, 0.13])),
    "index":  (np.array([0.07, 0.05, 0.12]), np.array([0.09, 0.05, 0.09])),
    "middle": (np.array([0.07, 0.02, 0.13]), np.array([0.09, 0.02, 0.10])),
    "ring":   (np.array([0.07, -0.01, 0.12]), np.array([0.09, -0.01, 0.09])),
    "pinky":  (np.array([0.07, -0.03, 0.11]), np.array([0.09, -0.03, 0.08])),
}
_FINGER_BASE = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


def landmarks_isaac(curl: float) -> np.ndarray:
    """21 x 3 wrist-local Isaac positions for a hand `curl` closed (0..1)."""
    L = np.zeros((21, 3))
    for f, base in _FINGER_BASE.items():
        aux = (1 - curl) * _OPEN[f][0] + curl * _CLOSED[f][0]
        tip = (1 - curl) * _OPEN[f][1] + curl * _CLOSED[f][1]
        # proximal..distal..tip: straight line from near the wrist to the tip
        root = np.array([0.0, aux[1], 0.03])
        for k in range(3):
            L[base + k] = root + (aux - root) * (k + 1) / 3.0
        L[base + 3] = tip
    return L


def isaac_to_unity_local(p: np.ndarray) -> np.ndarray:
    """Inverse of SkeletonProcessor: isaac = (-u_y, -u_x, u_z)  =>  unity = (-i_y, -i_x, i_z)."""
    return np.stack([-p[:, 1], -p[:, 0], p[:, 2]], axis=1)


def mirror_left(L_right_isaac: np.ndarray) -> np.ndarray:
    """Left hand = right hand mirrored across the thumb axis (Y) in the Isaac frame."""
    m = L_right_isaac.copy()
    m[:, 1] *= -1
    return m


def csv(label: str, values) -> str:
    return f"{label}:," + ",".join(f"{float(v):.5f}" for v in np.asarray(values).ravel())


def packet(t: float, amp: float, period: float, curl_amp: float) -> bytes:
    w = 2 * math.pi / period
    bob = amp * math.sin(w * t)
    curl = curl_amp * 0.5 * (1 - math.cos(w * t))
    # Unity world (LH, Y up, Z forward): head at eye height, hands ~0.4 m ahead.
    head = [0.0, 1.60, 0.0, 0.0, 0.0, 0.0, 1.0]
    right_w = [0.25, 1.10 + bob, 0.40, 0.0, 0.0, 0.0, 1.0]
    left_w = [-0.25, 1.10 - bob, 0.40, 0.0, 0.0, 0.0, 1.0]
    Lr = landmarks_isaac(curl)
    Ll = mirror_left(Lr)
    ident = np.tile([0.0, 0.0, 0.0, 1.0], (21, 1))
    lines = [
        csv("Right wrist", right_w),
        csv("Right landmarks", isaac_to_unity_local(Lr)),
        csv("Right rotations", ident),
        csv("Left wrist", left_w),
        csv("Left landmarks", isaac_to_unity_local(Ll)),
        csv("Left rotations", ident),
        csv("head", head),
    ]
    return ("\n".join(lines) + "\n").encode()


def _drain(sock: socket.socket):
    """vr_teleop sends `fb_*` feedback lines back on the same socket; discard them."""
    try:
        while True:
            if not sock.recv(4096):
                return
    except OSError:
        return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--amp", type=float, default=0.05, help="wrist bob amplitude (m, Unity y)")
    ap.add_argument("--period", type=float, default=6.0, help="bob / curl period (s)")
    ap.add_argument("--curl", type=float, default=1.0, help="finger curl amplitude (0 = hands stay open)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl+C)")
    args = ap.parse_args(argv)

    sock = socket.create_connection((args.host, args.port), timeout=5.0)
    sock.settimeout(None)
    threading.Thread(target=_drain, args=(sock,), daemon=True).start()
    print(f"fake_quest: streaming to {args.host}:{args.port} at {args.hz:.0f} Hz "
          f"(bob ±{args.amp} m, curl {args.curl}, period {args.period} s)")
    t0 = time.monotonic()
    period = 1.0 / args.hz
    n = 0
    try:
        while True:
            t = time.monotonic() - t0
            if args.seconds and t > args.seconds:
                break
            sock.sendall(packet(t, args.amp, args.period, args.curl))
            n += 1
            time.sleep(max(0.0, t0 + n * period - time.monotonic()))
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        sock.close()
    print(f"fake_quest: sent {n} packets")


if __name__ == "__main__":
    main()
