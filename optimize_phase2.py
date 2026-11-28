"""Search Phase 2 hyperparameters and evaluate the locked winners locally."""

import csv
import json
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np

import eval_harness as harness
from baseline_agents import QLearningReferenceAgent
from q_learning_agent import Agent as QLearning
from sarsa_agent import Agent as SARSA


DEVELOPMENT = range(200_000, 200_020)
HOLDOUT = range(300_000, 300_100)
AGENTS = {"Q-learning": QLearning, "SARSA": SARSA}
PREVIOUS = {
    "Q-learning": dict(alpha=0.05, gamma=0.95, start_epsilon=0.30,
                       end_epsilon=0.05, decay="linear"),
    "SARSA": dict(alpha=0.10, gamma=0.99, start_epsilon=0.50,
                  end_epsilon=0.05, decay="linear"),
}
GRID = {
    "alpha": (0.02, 0.05, 0.10),
    "gamma": (0.90, 0.95, 0.99),
    "start_epsilon": (0.10, 0.30, 0.50),
    "end_epsilon": (0.01, 0.05, 0.10),
    "decay": ("linear", "quadratic"),
}


class ObservedTradingEnv(harness.TradingEnv):
    """Count trades and invested days while the official harness scores."""

    def __init__(self, *args, **kwargs):
        self.episodes = []
        super().__init__(*args, **kwargs)

    def reset(self, seed=None):
        self.switches = 0
        self.invested_days = 0
        return super().reset(seed=seed)

    def step(self, action):
        previous_position = self.position
        outcome = super().step(action)
        self.switches += int(self.position != previous_position)
        self.invested_days += self.position
        return outcome

    def episode_summary(self):
        summary = super().episode_summary()
        score = self.portfolio_value - 1 - (
            harness.DRAWDOWN_PENALTY_LAMBDA * harness._max_drawdown(summary.portfolio_values)
        )
        self.episodes.append((score, self.switches, self.invested_days / self.episode_length))
        return summary


def tune():
    winners = {}
    rows = []
    with patch.multiple(harness, MDP_EVAL_SEED_START=DEVELOPMENT.start,
                        MDP_N_EVAL_EPISODES=len(DEVELOPMENT)):
        for name, agent in AGENTS.items():
            best_score = float("-inf")
            seen = {}
            for index, values in enumerate(product(*GRID.values()), start=1):
                settings = dict(zip(GRID, values))
                # Equal endpoints give a constant schedule for both decay modes.
                key = values[:-1] if values[2] == values[3] else values
                if key not in seen:
                    seen[key] = harness.run_mdp_eval(agent, settings)
                metrics = seen[key]
                rows.append(dict(algorithm=name, **settings, **metrics))
                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    winners[name] = settings
                if index % 27 == 0:
                    print(f"{name}: {index} combinations scored, best {best_score:.5f}",
                          flush=True)
    return winners, rows


def evaluate(agent, settings):
    environments = []

    def create_env(**kwargs):
        environment = ObservedTradingEnv(**kwargs)
        environments.append(environment)
        return environment

    with patch.object(harness, "TradingEnv", create_env):
        # The harness reads the dimensions from the same constructor it calls.
        create_env.N_STATES = ObservedTradingEnv.N_STATES
        create_env.N_ACTIONS = ObservedTradingEnv.N_ACTIONS
        metrics = harness.run_mdp_eval(agent, settings)
    scores, switches, invested = np.asarray(environments[0].episodes).T
    np.testing.assert_allclose(scores.mean(), metrics["score"], atol=1e-12)
    return dict(**metrics, position_changes=float(switches.mean()),
                invested_fraction=float(invested.mean())), scores


def paired_difference(left, right, bootstrap_indices):
    differences = left - right
    lower, upper = np.quantile(differences[bootstrap_indices].mean(axis=1), [0.025, 0.975])
    return dict(difference=float(differences.mean()), ci95=[float(lower), float(upper)])


def main():
    winners, rows = tune()
    with Path("phase2_grid_results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Locked winners: {winners}", flush=True)
    candidates = {"Provided baseline": (QLearningReferenceAgent, {})}
    for name, agent in AGENTS.items():
        candidates[f"Previous {name}"] = (agent, PREVIOUS[name])
        candidates[f"Tuned {name}"] = (agent, winners[name])
    candidates["Q-learning at tuned SARSA settings"] = (QLearning, winners["SARSA"])
    candidates["SARSA at tuned Q-learning settings"] = (SARSA, winners["Q-learning"])
    with patch.multiple(harness, MDP_EVAL_SEED_START=HOLDOUT.start,
                        MDP_N_EVAL_EPISODES=len(HOLDOUT)):
        holdout_results = {name: evaluate(*candidate) for name, candidate in candidates.items()}
    official = {name: evaluate(*candidate)[0] for name, candidate in candidates.items()}

    # Pair results by seed to remove some of the variation due to market paths.
    indices = np.random.default_rng(42).integers(len(HOLDOUT), size=(10_000, len(HOLDOUT)))
    comparisons = [
        ("Tuned Q-learning", "Previous Q-learning"),
        ("Tuned SARSA", "Previous SARSA"),
        ("Tuned Q-learning", "Provided baseline"),
        ("Tuned SARSA", "Provided baseline"),
        ("Tuned Q-learning", "Tuned SARSA"),
        ("Tuned Q-learning", "SARSA at tuned Q-learning settings"),
        ("Q-learning at tuned SARSA settings", "Tuned SARSA"),
    ]
    summary = {
        "settings": {name: settings for name, (_, settings) in candidates.items()},
        "holdout": {name: metrics for name, (metrics, _) in holdout_results.items()},
        "official": official,
        "holdout_differences": {
            f"{left} minus {right}": paired_difference(
                holdout_results[left][1], holdout_results[right][1], indices
            ) for left, right in comparisons
        },
    }
    Path("phase2_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, metrics in summary["holdout"].items():
        print(f"{name}: holdout {metrics['score']:.5f}, official {official[name]['score']:.5f}")


if __name__ == "__main__":
    main()
