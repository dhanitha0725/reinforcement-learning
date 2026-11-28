"""A bandit that gives recent strategy rewards more weight."""

import numpy as np

from agent_base import BanditAgent


class Agent(BanditAgent):
    """Explore with epsilon-greedy and update using a constant learning rate."""

    def __init__(self, n_arms: int, *, epsilon: float = 0.02, alpha: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.alpha = alpha
        self.values = np.zeros(n_arms)
        self._rng = np.random.default_rng()
        self._untried_arms = self._rng.permutation(n_arms).tolist()

    def select_arm(self) -> int:
        if self._untried_arms:
            return self._untried_arms.pop()
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.values[arm] += self.alpha * (reward - self.values[arm])
