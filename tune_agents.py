"""Compare agent settings on development markets using the starter scorer."""

from itertools import product
from unittest.mock import patch

import eval_harness
from bandit_agent import Agent as BanditAgent
from baseline_agents import QLearningReferenceAgent
from q_learning_agent import Agent as QLearningAgent
from sarsa_agent import Agent as SarsaAgent


DEVELOPMENT_SEEDS = range(200_000, 200_020)


def compare_bandit_settings() -> None:
    # Limit the seed override to this experiment. Official tests use the
    # starter harness directly through test_locally.py.
    with patch.object(eval_harness, "BANDIT_EVAL_SEEDS", DEVELOPMENT_SEEDS):
        for epsilon, alpha in product((0.10, 0.15, 0.25), (0.1, 0.2)):
            settings = {"epsilon": epsilon, "alpha": alpha}
            result = eval_harness.run_bandit_eval(BanditAgent, settings)
            print(f"Bandit {settings} score={result['score']:.8f}", flush=True)


def compare_trading_settings() -> None:
    with patch.multiple(eval_harness, MDP_EVAL_SEED_START=DEVELOPMENT_SEEDS.start,
                        MDP_N_EVAL_EPISODES=len(DEVELOPMENT_SEEDS)):
        baseline = eval_harness.run_mdp_eval(QLearningReferenceAgent)
        print(f"Q-learning reference score={baseline['score']:.8f}", flush=True)
        for alpha, start_epsilon, gamma in product((0.1, 0.05, 0.2), (0.5, 0.3), (0.99, 0.95)):
            settings = dict(alpha=alpha, start_epsilon=start_epsilon, gamma=gamma,
                            end_epsilon=0.05, decay="linear")
            for name, agent in (("Q-learning", QLearningAgent), ("SARSA", SarsaAgent)):
                result = eval_harness.run_mdp_eval(agent, settings)
                print(f"{name} {settings} score={result['score']:.8f} "
                      f"return={result['mean_return']:.8f} "
                      f"drawdown={result['mean_max_drawdown']:.8f}", flush=True)


if __name__ == "__main__":
    compare_bandit_settings()
    compare_trading_settings()
