"""Bring-up step 1 (friday_setup_plan.md section 7): teleop-only, no learner.

Policy action = zeros. Hold GRIP on a Quest controller to clutch that arm to
your hand through the position-error coupling; release to freeze. ESC (env
listener) ends the episode; Ctrl+C quits.

Run AFTER friday_server.py is up and the state topics are publishing:
    cd examples/experiments/rby1_reach
    python teleop_test.py
"""

import pathlib
import sys

import numpy as np

# Make `experiments.*` importable when run from this directory (the other
# example scripts live in examples/ and get it via sys.path[0]).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.rby1_reach.config import TrainConfig  # noqa: E402


from experiments.rby1_reach.config import LeftArmConfig, RightArmConfig


def main():
    print("reset() will DRIVE BOTH ARMS to RESET_POSE:")
    print(f"  left : {LeftArmConfig.RESET_POSE}")
    print(f"  right: {RightArmConfig.RESET_POSE}")
    input("Verify these are measured (not placeholders), e-stop in reach, then press Enter...")
    env = TrainConfig().get_environment(fake_env=False, classifier=False)
    obs, _ = env.reset()
    print("teleop_test: hold GRIP to drive an arm; ESC = episode end; Ctrl+C = quit")

    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    step = 0
    try:
        while True:
            obs, rew, done, truncated, info = env.step(zero)
            step += 1
            if step % 10 == 0:  # ~1 Hz at hz=10
                mark = "INTERVENING" if "intervene_action" in info else "policy(zero)"
                print(f"step {step:5d}  {mark}  A={info.get('right')} X={info.get('left')}")
            # Episodes end ONLY when the operator presses X (left controller)
            # — QuestIntervention raises `truncated` on the rising edge. The
            # length limit is parked at 100000 so nothing ends on its own.
            if done or truncated:
                print("episode end (X pressed) — resetting to RESET_POSE")
                obs, _ = env.reset()
                step = 0
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
