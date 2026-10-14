#!/usr/bin/env python3
"""Hand-mode chain check: Quest (or fake_quest) -> vr_teleop -> ZmqQuestReader
-> env -> rby1_server -> controller. Prints a live table and a verdict.

Run (conda hilserl) with the stack up (sil.sh up/enable/ready, rby1_server,
run_vr_teleop.sh, headset streaming or fake_quest.py):

    cd examples/experiments/rby1_reach
    RBY1_INPUT=hand python ../../../serl_robot_infra/franka_env/vr_hand_teleop/chain_check.py --seconds 60

Each row: wire (what the bridge publishes: wrist z, index_joint1), robot (what
the server measures: EE z, hand_q index_joint1), and whether the env is
intervening on that side. Verdict checks per side: wire seen, intervention
engaged, EE moved with the wrist, fingers followed the wire, no finger hold
longer than --max-hold while the wire kept moving.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

import numpy as np

os.environ.setdefault("RBY1_INPUT", "hand")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "examples"))

from experiments.rby1_reach.config import INPUT_MODE, LeftArmConfig, RightArmConfig, TrainConfig  # noqa: E402

SIDES = ("left", "right")
URLS = {"left": LeftArmConfig.SERVER_URL, "right": RightArmConfig.SERVER_URL}


def getstate(side):
    req = urllib.request.Request(URLS[side] + "getstate", method="POST")
    return json.load(urllib.request.urlopen(req, timeout=1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-hold", type=float, default=2.0, help="s; a finger frozen longer while the wire moves = fail")
    ap.add_argument("--csv", default="", help="write every step here")
    ap.add_argument("--press-x-at", type=float, default=0.0,
                    help="s; inject a keyboard `x` (pynput) then — must print 'episode end -> reset'")
    args = ap.parse_args()
    if INPUT_MODE != "hand":
        sys.exit("set RBY1_INPUT=hand")

    env = TrainConfig().get_environment(fake_env=False, classifier=False)
    e = env
    while not hasattr(e, "reader"):   # walk the wrapper stack down to XHandPassthrough / QuestIntervention
        e = e.env
    reader = e.reader
    obs, _ = env.reset()
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    rows = []
    episode_ends = 0
    reset_times = []
    pressed = False
    t0 = time.monotonic()
    last_print = 0.0
    print(f"{'t':>5} | {'wireLz':>6} {'wireLi':>6} {'robLz':>6} {'robLi':>6} L | {'wireRz':>6} {'wireRi':>6} {'robRz':>6} {'robRi':>6} R")
    try:
        while time.monotonic() - t0 < args.seconds:
            obs, rew, done, truncated, info = env.step(zero)
            t = time.monotonic() - t0
            poses, fingers = reader.poses(), reader.fingers()
            row = {"t": t, "intervene": "intervene_action" in info}
            for side in SIDES:
                st = getstate(side)
                row[f"wire_{side}_z"] = poses[side][0][2] if side in poses else np.nan
                row[f"wire_{side}_i"] = fingers[side][4] if side in fingers else np.nan
                row[f"rob_{side}_z"] = st["pose"][2]
                row[f"rob_{side}_i"] = st["hand_q"][4]
                row[f"eng_{side}"] = side in poses
            rows.append(row)
            if t - last_print >= 2.0:
                last_print = t
                print(f"{t:5.1f} | " + " | ".join(
                    f"{row[f'wire_{s}_z']:6.3f} {row[f'wire_{s}_i']:6.2f} {row[f'rob_{s}_z']:6.3f} {row[f'rob_{s}_i']:6.2f} {'*' if row[f'eng_{s}'] else '-'}"
                    for s in SIDES))
            if args.press_x_at and not pressed and t >= args.press_x_at:
                pressed = True
                from pynput.keyboard import Controller
                Controller().tap("x")
                print(f"{t:5.1f} | injected key x")
            if done or truncated:
                episode_ends += 1
                print(f"{t:5.1f} | episode end -> reset")
                obs, _ = env.reset()
                reset_times.append(time.monotonic() - t0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---- verdict ----------------------------------------------------------
    # Rows inside the reset transient (arms driving back to READY, re-latch) are
    # not tracking and are left out of the tracking metrics.
    ts_all = np.array([r["t"] for r in rows])
    settled = np.ones(len(rows), dtype=bool)
    for tr in reset_times:
        settled &= ~((ts_all >= tr - 1.0) & (ts_all < tr + 6.0))
    rows = [r for r, keep in zip(rows, settled) if keep]
    ok_all = True
    for side in SIDES:
        wz = np.array([r[f"wire_{side}_z"] for r in rows]); wi = np.array([r[f"wire_{side}_i"] for r in rows])
        rz = np.array([r[f"rob_{side}_z"] for r in rows]); ri = np.array([r[f"rob_{side}_i"] for r in rows])
        ts = np.array([r["t"] for r in rows]); eng = np.array([r[f"eng_{side}"] for r in rows])
        seen = np.isfinite(wz).mean()
        moved = np.nanmax(rz) - np.nanmin(rz) if len(rz) else 0.0
        wire_moved = np.nanmax(wz) - np.nanmin(wz) if np.isfinite(wz).any() else 0.0
        fin_range = np.nanmax(ri) - np.nanmin(ri)
        # Lag-tolerant tracking score: best correlation of EE z against the wire
        # z shifted by 0..1.5 s (fabrics + 10 Hz env + exaggeration add ~0.5 s).
        corr_z, lag_best = np.nan, 0.0
        if np.isfinite(wz).sum() > 5 and wire_moved > 1e-3:
            for lag in np.arange(0.0, 1.55, 0.1):
                wz_shift = np.interp(ts - lag, ts, np.nan_to_num(wz, nan=np.nanmean(wz)))
                c = np.corrcoef(wz_shift, rz)[0, 1]
                if np.isnan(corr_z) or c > corr_z:
                    corr_z, lag_best = c, lag
        # longest stretch where the robot finger did not change while the wire did
        hold, worst = 0.0, 0.0
        for k in range(1, len(rows)):
            if np.isfinite(wi[k]) and abs(wi[k] - wi[k - 1]) > 0.01 and abs(ri[k] - ri[k - 1]) < 1e-4:
                hold += ts[k] - ts[k - 1]
                worst = max(worst, hold)
            else:
                hold = 0.0
        ok = seen > 0.9 and eng.mean() > 0.9 and moved > 0.02 and fin_range > 0.2 and worst <= args.max_hold and (np.isnan(corr_z) or corr_z > 0.75)
        ok_all &= ok
        print(f"{side:5s}: wire seen {seen:4.0%}  engaged {eng.mean():4.0%}  EE z range {moved:.3f} m  "
              f"corr(wire z, EE z) {corr_z:.2f} @ lag {lag_best:.1f} s  finger range {fin_range:.2f} rad  "
              f"worst finger hold {worst:.1f} s  -> {'OK' if ok else 'FAIL'}")
    if args.press_x_at:
        x_ok = episode_ends >= 1
        ok_all &= x_ok
        print(f"key x -> episode end: {'OK' if x_ok else 'FAIL'} ({episode_ends} episode end(s))")
    print("CHAIN OK" if ok_all else "CHAIN FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
