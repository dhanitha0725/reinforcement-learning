"""Bayesian optimization of epsilon-greedy using a Gaussian process.

Requires NumPy and SciPy. The submission agent itself only needs NumPy.
"""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr

import eval_harness
from bandit_agent import Agent
from baseline_agents import RandomBanditAgent, EpsilonGreedyBanditAgent
from ucb_bandit_agent import Agent as UCB


DEVELOPMENT_SEEDS = range(700_000, 700_030)
TEST_SEEDS = range(800_000, 800_100)
TRIALS = 48
INITIAL_TRIALS = 10
EPSILON_RANGE = (0.0, 0.50)
ALPHA_RANGE = (0.01, 1.0)


def decode(point):
    """Scale epsilon linearly and alpha logarithmically."""
    epsilon = EPSILON_RANGE[0] + point[0] * (EPSILON_RANGE[1] - EPSILON_RANGE[0])
    alpha = np.exp(np.log(ALPHA_RANGE[0]) + point[1] * np.log(ALPHA_RANGE[1] / ALPHA_RANGE[0]))
    return {"epsilon": float(epsilon), "alpha": float(alpha)}


def evaluate(agent, settings, seeds):
    with patch.object(eval_harness, "BANDIT_EVAL_SEEDS", seeds):
        result = eval_harness.run_bandit_eval(agent, settings)
    return np.asarray(result["per_seed_rewards"], dtype=float)


def kernel(left, right):
    """Matern 5/2 covariance on normalized parameters, with fixed length scale."""
    distance = np.linalg.norm((left[:, None, :] - right[None, :, :]) / 0.3, axis=2)
    scaled = np.sqrt(5.0) * distance
    return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def expected_improvement(points, scores, options):
    """Fit a GP and calculate expected improvement for each candidate."""
    mean, scale = scores.mean(), max(scores.std(), 0.05)
    targets = (scores - mean) / scale
    covariance = kernel(points, points) + 0.2**2 * np.eye(len(points))
    chol = np.linalg.cholesky(covariance)
    cross = kernel(points, options)
    weights = solve_triangular(chol.T, solve_triangular(chol, targets, lower=True))
    prediction = cross.T @ weights
    projected = solve_triangular(chol, cross, lower=True)
    sigma = np.sqrt(np.maximum(1.0 - (projected**2).sum(axis=0), 1e-12))
    improvement = prediction - targets.max() - 0.01
    z = improvement / sigma
    return improvement * ndtr(z) + sigma * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def main():
    # Explicit seed keeps optimizer randomness independent of the harness patch.
    rng = np.random.default_rng(20261128)
    points, scores, records = [], [], []
    for trial in range(TRIALS):
        if trial < INITIAL_TRIALS:
            point = rng.random(2)
            source = "initial random"
        else:
            options = rng.random((2048, 2))
            acquisition = expected_improvement(np.asarray(points), np.asarray(scores), options)
            point = options[np.argmax(acquisition)]
            source = "GP expected improvement"
        settings = decode(point)
        rewards = evaluate(Agent, settings, DEVELOPMENT_SEEDS)
        points.append(point)
        scores.append(float(rewards.mean()))
        records.append(dict(trial=trial + 1, method=source, **settings,
                            development_score=float(rewards.mean()),
                            development_std=float(rewards.std(ddof=1))))
        print(f"Trial {trial + 1}/{TRIALS}: epsilon={settings['epsilon']:.4f} "
              f"alpha={settings['alpha']:.4f} score={scores[-1]:.4f} "
              f"best={max(scores):.4f}", flush=True)

    with Path("bandit_gp_trials.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    best = records[int(np.argmax(scores))]
    candidates = {
        "GP candidate": (Agent, {"epsilon": best["epsilon"], "alpha": best["alpha"]}),
        "Previous broad search": (Agent, {"epsilon": 0.02, "alpha": 0.1}),
        "Previous small grid": (Agent, {"epsilon": 0.15, "alpha": 0.2}),
        "UCB": (UCB, {}),
        "Random baseline": (RandomBanditAgent, {}),
        "Reference epsilon-greedy": (EpsilonGreedyBanditAgent, {}),
    }
    result = {"development_seeds": [DEVELOPMENT_SEEDS.start, DEVELOPMENT_SEEDS.stop - 1],
              "test_seeds": [TEST_SEEDS.start, TEST_SEEDS.stop - 1],
              "best_development_trial": best, "candidates": {}}
    for name, (agent, settings) in candidates.items():
        test = evaluate(agent, settings, TEST_SEEDS)
        official = evaluate(agent, settings, [7, 17, 27, 37, 47])
        result["candidates"][name] = {
            "settings": settings, "test_mean": float(test.mean()),
            "test_std": float(test.std(ddof=1)), "test_rewards": test.tolist(),
            "official_mean": float(official.mean()), "official_rewards": official.tolist(),
        }
        print(f"{name}: fresh={test.mean():.5f}, official={official.mean():.5f}", flush=True)
    Path("bandit_gp_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
