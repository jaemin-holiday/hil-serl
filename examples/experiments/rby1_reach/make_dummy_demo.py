"""Make a schema-valid dummy demo pkl so the learner can start BEFORE any
real demos exist (train_rlpd's learner hard-asserts --demo_path).

Transitions are random samples from the fake env's own spaces — worthless for
learning, correct in shape. Use ONLY for the issue-2 plumbing test
(intervention -> both buffers); replace with record_demos.py output for real
training.

Run from this directory (no robot / headset / ROS needed):
    python make_dummy_demo.py            # -> demo_data/dummy_demo_10.pkl
"""

import os
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.friday_reach.config import TrainConfig  # noqa: E402

N = 10

env = TrainConfig().get_environment(fake_env=True, classifier=False)
transitions = []
for _ in range(N):
    obs = env.observation_space.sample()
    next_obs = env.observation_space.sample()
    transitions.append(dict(
        observations=obs,
        actions=env.action_space.sample(),
        next_observations=next_obs,
        rewards=0.0,
        masks=1.0,
        dones=False,
    ))

os.makedirs("demo_data", exist_ok=True)
out = f"demo_data/dummy_demo_{N}.pkl"
with open(out, "wb") as f:
    pickle.dump(transitions, f)
print(f"wrote {N} dummy transitions -> {out}")
