"""Reproduce the wider Phase 1 grid and check fixed candidates locally."""

import csv
from itertools import product
from pathlib import Path

import numpy as np

from bandit_agent import Agent
from compare_bandits import evaluate
from thompson_bandit_agent import Agent as Thompson
from ucb_bandit_agent import Agent as UCB


DEVELOPMENT_SEEDS = range(400_000, 400_030)
CONFIRMATION_SEEDS = range(900_000, 900_300)
EXPLORATION_RATES = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
LEARNING_RATES = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


def write_rows(filename, rows):
    with Path(filename).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    grid = []
    for epsilon, alpha in product(EXPLORATION_RATES, LEARNING_RATES):
        settings = dict(epsilon=epsilon, alpha=alpha)
        score = evaluate(Agent, settings, DEVELOPMENT_SEEDS)["score"]
        grid.append(dict(**settings, score=score))
    write_rows("bandit_wider_grid_replay.csv", grid)
    winner = max(grid, key=lambda row: row["score"])
    print("Reproduced grid winner", winner, flush=True)

    candidates = {
        "Wider-grid epsilon-greedy": (Agent, dict(epsilon=0.02, alpha=0.10)),
        "Original epsilon-greedy": (Agent, dict(epsilon=0.15, alpha=0.20)),
        "GP-selected epsilon-greedy": (Agent, dict(epsilon=0.0003117774638627946,
                                                    alpha=0.11434624300384755)),
        "Sliding-window UCB": (UCB, dict(window=50, exploration=0.01)),
        "Discounted Thompson": (Thompson, dict(discount=0.94, noise=0.01)),
    }
    rewards = {
        name: np.asarray(evaluate(agent, settings, CONFIRMATION_SEEDS)["per_seed_rewards"])
        for name, (agent, settings) in candidates.items()
    }
    write_rows("bandit_final_check.csv", [
        dict(seed=seed, **{name: values[index] for name, values in rewards.items()})
        for index, seed in enumerate(CONFIRMATION_SEEDS)
    ])
    rng = np.random.default_rng(20261128)
    indices = rng.integers(len(CONFIRMATION_SEEDS), size=(10_000, len(CONFIRMATION_SEEDS)))
    selected = rewards["Wider-grid epsilon-greedy"]
    for name, values in rewards.items():
        difference = selected - values
        lower, upper = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
        print(f"{name}: mean={values.mean():.5f}, selected minus this={difference.mean():+.5f}, "
              f"95% interval [{lower:+.5f}, {upper:+.5f}]")


if __name__ == "__main__":
    main()
