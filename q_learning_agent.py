"""Tabular Q-learning for the trading simulator."""

import numpy as np

from agent_base import MDPAgent


class Agent(MDPAgent):
    """Learn from the highest estimated value in the next state."""

    TRAINING_EPISODES = 300

    def __init__(self, n_states: int, n_actions: int, *, alpha: float = 0.02,
                 gamma: float = 0.99, start_epsilon: float = 0.5,
                 end_epsilon: float = 0.1, decay: str = "linear"):
        if not 0 <= end_epsilon <= start_epsilon <= 1:
            raise ValueError("Expected 0 <= end_epsilon <= start_epsilon <= 1")
        if decay not in ("linear", "quadratic"):
            raise ValueError("decay must be 'linear' or 'quadratic'")
        super().__init__(n_states, n_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay = decay
        self.Q = np.zeros((n_states, n_actions))
        self._rng = np.random.default_rng()
        self._episode_index = -1
        self._epsilon = start_epsilon
        self._is_eval = False

    def start_episode(self) -> None:
        self._episode_index += 1
        progress = min(self._episode_index / (self.TRAINING_EPISODES - 1), 1.0)
        if self.decay == "quadratic":
            progress = 1.0 - (1.0 - progress) ** 2
        self._epsilon = (1.0 - progress) * self.start_epsilon + progress * self.end_epsilon

    def set_eval_mode(self, is_eval: bool) -> None:
        self._is_eval = is_eval

    def act(self, state: int) -> int:
        if not self._is_eval and self._rng.random() < self._epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int | None, done: bool) -> None:
        if self._is_eval:
            return
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state].max()
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
