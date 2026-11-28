"""Discounted Gaussian Thompson sampling for changing rewards."""

import numpy as np
from agent_base import BanditAgent


class Agent(BanditAgent):
    def __init__(self, n_arms, *, discount=0.94, noise=0.01, prior_std=0.05):
        super().__init__(n_arms)
        self.discount = discount
        self.noise = noise
        self.prior_std = prior_std
        self._weights = np.zeros(n_arms)
        self._totals = np.zeros(n_arms)
        self._rng = np.random.default_rng()
        self._untried = self._rng.permutation(n_arms).tolist()

    def select_arm(self):
        if self._untried:
            return int(self._untried.pop())
        # A zero-mean Gaussian prior plus discounted Gaussian observations.
        precision = 1 / self.prior_std ** 2 + self._weights / self.noise ** 2
        means = (self._totals / self.noise ** 2) / precision
        samples = self._rng.normal(means, np.sqrt(1 / precision))
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self._weights *= self.discount
        self._totals *= self.discount
        self._weights[arm] += 1
        self._totals[arm] += reward
