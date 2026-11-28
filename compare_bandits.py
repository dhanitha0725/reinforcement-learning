"""Tune three bandit algorithms and compare them on new market seeds."""

from itertools import product
from unittest.mock import patch

import numpy as np
import eval_harness
from bandit_agent import Agent as EpsilonGreedy
from ucb_bandit_agent import Agent as UCB
from thompson_bandit_agent import Agent as Thompson


DEVELOPMENT_SEEDS = range(200_000, 200_020)
COMPARISON_SEEDS = range(300_000, 300_020)


def evaluate(agent, settings, seeds):
    with patch.object(eval_harness, "BANDIT_EVAL_SEEDS", seeds):
        return eval_harness.run_bandit_eval(agent, settings)


def main():
    candidates = [
        ("Epsilon-greedy", EpsilonGreedy,
         [dict(epsilon=e, alpha=a) for e, a in product((0.1, 0.15, 0.25), (0.1, 0.2))]),
        ("Sliding-window UCB", UCB,
         [dict(window=w, exploration=c) for w, c in product((30, 50, 80), (0.01, 0.03, 0.06))]),
        ("Discounted Thompson", Thompson,
         [dict(discount=d, noise=n) for d, n in product((0.94, 0.97, 0.99), (0.01, 0.03, 0.06))]),
    ]
    print("DEVELOPMENT SWEEP", flush=True)
    chosen = []
    for name, agent, grid in candidates:
        scored = []
        for settings in grid:
            score = evaluate(agent, settings, DEVELOPMENT_SEEDS)["score"]
            scored.append((score, settings))
            print(f"{name}: {settings} mean={score:.6f}", flush=True)
        score, settings = max(scored, key=lambda result: result[0])
        chosen.append((name, agent, settings))
        print(f"Chosen {name}: {settings}, mean={score:.6f}", flush=True)

    print("\nINDEPENDENT COMPARISON", flush=True)
    for name, agent, settings in chosen:
        result = evaluate(agent, settings, COMPARISON_SEEDS)
        rewards = np.asarray(result["per_seed_rewards"])
        print(f"{name}: {settings} mean={rewards.mean():.6f} "
              f"min={rewards.min():.6f} max={rewards.max():.6f} "
              f"std={rewards.std():.6f}", flush=True)
        for seed, reward in zip(COMPARISON_SEEDS, rewards):
            print(f"  {seed}: {reward:.6f}", flush=True)


if __name__ == "__main__":
    main()
