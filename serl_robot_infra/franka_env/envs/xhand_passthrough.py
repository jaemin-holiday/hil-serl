"""Forward the operator's finger targets to the robot server, outside the
policy action.

Hand mode only: the action space stays the 12D dual-arm one, fingers are
operator-driven through POST /<side>/hand (robot_servers/rby1_server.py) and
recorded in info["fingers"] so demos carry them. Whether fingers later enter
the action space is a separate decision; this wrapper is the no-learning path.
"""

import logging

import gymnasium as gym
import numpy as np
import requests

logger = logging.getLogger(__name__)
SIDES = ("left", "right")


class XHandPassthrough(gym.Wrapper):
    def __init__(self, env, reader, min_delta_rad: float = 0.005, timeout_s: float = 0.2):
        super().__init__(env)
        self.reader = reader
        self.min_delta = float(min_delta_rad)
        self.timeout = float(timeout_s)
        self._urls = {
            "left": env.unwrapped.env_left.url,
            "right": env.unwrapped.env_right.url,
        }
        self._last_sent = {s: None for s in SIDES}
        self._warned = {s: False for s in SIDES}

    def _send(self, side, q):
        last = self._last_sent[side]
        if last is not None and np.max(np.abs(q - last)) < self.min_delta:
            return
        try:
            requests.post(self._urls[side] + "hand", json={"q": q.tolist()}, timeout=self.timeout)
            self._last_sent[side] = q.copy()
        except requests.RequestException as e:
            if not self._warned[side]:
                logger.warning("XHandPassthrough: %s hand POST failed (%s); will keep trying silently", side, e)
                self._warned[side] = True

    def step(self, action):
        fingers = self.reader.fingers()
        for side, q in fingers.items():
            self._send(side, q)
        obs, rew, done, truncated, info = self.env.step(action)
        if fingers:
            info["fingers"] = {s: q.copy() for s, q in fingers.items()}
        return obs, rew, done, truncated, info
