# RL Trading Arena implementation results

These are the original Phase 2 results. The current defaults and their evaluation are documented in [Phase 2 hyperparameter optimization](PHASE2_OPTIMIZATION.md).

Run these commands from the assignment folder:

```powershell
python -B tune_agents.py
python -B test_locally.py bandit_agent.py bandit
python -B test_locally.py q_learning_agent.py mdp
python -B test_locally.py sarsa_agent.py mdp
```

## Development comparison

The sweep used simulator seeds `200000–200019` for scoring candidates. Both trading agents trained on the same 300 episodes. At the original shared settings (`alpha=0.1`, `gamma=0.99`, starting exploration `0.5`), Q-learning scored `-0.00261` and SARSA scored `0.01445` on development markets. The sweep also tried learning rates `0.05` and `0.2` for both trading agents. Defaults were selected by development score before running the official tests.

| Agent | Selected settings | Development score |
| --- | --- | ---: |
| Bandit | Exploration `0.15`, learning rate `0.2` | 1.21256 |
| Q-learning | Exploration `0.30 → 0.05`, learning rate `0.05`, discount `0.95` | 0.01209 |
| SARSA | Exploration `0.50 → 0.05`, learning rate `0.1`, discount `0.99` | 0.01445 |

## Official local comparison

Each command produced the same result on two runs. The bandit score is average cumulative reward. Trading score is average return minus half of average maximum drawdown. Drawdown measures the largest fall from a previous portfolio peak in an episode.

| Agent | Score | Best bundled baseline | Difference | Average return | Average max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bandit | 1.1077 | 0.3375 | +0.7702 | — | — |
| Q-learning | 0.0216 | 0.0279 | -0.0063 | 3.49% | 2.66% |
| SARSA | 0.0298 | 0.0279 | +0.0019 | 4.53% | 3.08% |

The Phase 2 results above use each agent's selected settings, so the score gap reflects both the algorithms and their settings.

## Phase 1 algorithm comparison

Run `python -B compare_bandits.py` to reproduce the sweep and all individual results in `bandit_comparison_results.txt`. The sweep chooses settings using seeds 200000–200019, and then compares the chosen settings on separate seeds 300000–300019. Each run contains 200 pulls. Higher average cumulative reward is better.

| Method | Chosen settings | Development average | New-seed average | New-seed range |
| --- | --- | ---: | ---: | ---: |
| Epsilon-greedy | epsilon 0.15, alpha 0.2 | 1.2126 | 1.2270 | 0.1039 to 2.3425 |
| Sliding-window UCB | window 50, exploration 0.01 | 1.2215 | 1.3678 | 0.4104 to 2.2369 |
| Discounted Gaussian Thompson sampling | discount 0.94, noise 0.01, prior standard deviation 0.05 | 0.9224 | 0.9789 | -0.0488 to 1.5957 |

UCB is the selected candidate because it has the highest average on the separate comparison seeds. The official local test for `ucb_bandit_agent.py` scored 0.9759, compared with 0.3375 for the random baseline. The previously tested epsilon-greedy agent scored 1.1077 on those same five official seeds. This shows that the ranking varies between seed sets; the comparison does not establish a universal winner. The official seeds were already inspected for epsilon-greedy, so they are not an untouched test set.

The environment generates rewards sequentially, and its random strategy consumes extra random numbers. Therefore, identical seeds do not guarantee identical market trajectories after the agents choose different arms. Both tests nevertheless follow the starter harness rules. Different grid sizes and the limited set of values also mean this is a comparison of these implementations and settings, not every possible version of the algorithms.
