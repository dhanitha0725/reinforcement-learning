"""Sliding-window upper confidence bound for changing strategy rewards."""

from collections import deque
import numpy as np
from agent_base import BanditAgent


class Agent(BanditAgent):
    def __init__(self, n_arms, *, window=50, exploration=0.01):
        super().__init__(n_arms)
        self.window = window
        self.exploration = exploration
        self._history = deque(maxlen=window)
        self._rng = np.random.default_rng()
        self._untried = self._rng.permutation(n_arms).tolist()

    def select_arm(self):
        if self._untried:
            return int(self._untried.pop())
        counts = np.zeros(self.n_arms)
        totals = np.zeros(self.n_arms)
        for arm, reward in self._history:
            counts[arm] += 1
            totals[arm] += reward
        missing = np.flatnonzero(counts == 0)
        if len(missing):
            return int(self._rng.choice(missing))
        confidence = self.exploration * np.sqrt(np.log(len(self._history)) / counts)
        return int(np.argmax(totals / counts + confidence))

    def update(self, arm, reward):
        self._history.append((arm, reward))
