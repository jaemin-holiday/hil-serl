"""Run remy2's vr_teleop for the RB-Y1 + XHand without touching remy2.

Quest hand tracking -> remy2 vr_teleop (wrist pose + 12 XHand joints per side,
IK) -> ZMQ :5555 (default) or ROS 2 holiday/manipulation (--publisher ros2).

Everything rby1-specific comes from this directory; remy2 is a pinned,
read-only checkout:
  - the XHand hand profile is injected into remy2's registry (xhand_profile.py)
  - vr_teleop's `--robot` lookup is replaced by rby1_xhand.robot.yaml here
  - `--ik-config xhand_right.yaml` and `--config rby1_xhand.routing.yaml` here
    are passed unless given on the command line

Run through run_vr_teleop.sh (remy2's venv: py3.12, torch-cpu, pinocchio, nlopt).
Any other vr_teleop flag passes through (`--publisher ros2`, `--retargeter noop`,
`--record ...`). Keyboard: wr/wl wrist lock, fr/fl finger lock, Ctrl+C.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMY2 = Path(os.environ.get("REMY2", Path.home() / "workspace" / "remy2")).resolve()
VR_TELEOP = REMY2 / "vr_teleop"

if not (VR_TELEOP / "main.py").is_file():
    sys.exit(f"remy2 vr_teleop not found at {VR_TELEOP} (set REMY2=/path/to/remy2)")
sys.path.insert(0, str(VR_TELEOP))  # main.py's own imports are vr_teleop-relative

import main as vr_main  # noqa: E402  (remy2/vr_teleop/main.py)

import xhand_profile  # noqa: E402

# One real check that the remy2 we found still has the seams we use.
for attr in ("resolve_robot_profile", "main", "_build_parser"):
    if not hasattr(vr_main, attr):
        sys.exit(f"remy2 vr_teleop/main.py has no {attr}: pin remy2 to a known commit")

vr_main.resolve_robot_profile = lambda args: xhand_profile.load_robot_profile()


def _default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv += [flag, value]


# `--robot` only has to pass argparse's choices; resolve_robot_profile is ours.
_default("--robot", "friday_gen1_mk3")
_default("--ik-config", str(xhand_profile.IK_CONFIG))
_default("--config", str(xhand_profile.ROUTING_YAML))

if __name__ == "__main__":
    vr_main.main()
