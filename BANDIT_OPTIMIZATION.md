# Phase 1 optimization history

All scores below are average cumulative reward per 200-pull run. Higher is better. The official local seeds are 7, 17, 27, 37, and 47. The leaderboard has not been contacted.

| Experiment | Selected exploration | Learning rate | Development | Fresh comparison | Official local |
|---|---:|---:|---:|---:|---:|
| Original grid | 0.15 | 0.20 | 1.21256 (20 seeds) | 1.2270 (20 seeds) | 1.10774 |
| Wider grid | 0.02 | 0.10 | 1.46708 (30 seeds) | 1.28436 (100 seeds) | 1.31772 |
| Gaussian-process search | 0.0003 | 0.1143 | 1.31550 (30 seeds) | 1.24581 (100 seeds) | 1.12631 |

The original grid's six results and the comparison among bandit methods are preserved in `bandit_comparison_results.txt`. The earlier wider-grid run printed only its winner and summary. A deterministic replay reproduced the original winning score and saved all 48 scores in [bandit_wider_grid_replay.csv](bandit_wider_grid_replay.csv). The fresh-test seeds in the above rows differ across experiments.

## Gaussian-process search

Run `python -B optimize_bandit_gp.py`. It needs SciPy as well as NumPy. Only the experiment script uses SciPy, and the submitted agent requires NumPy alone.

The script uses 10 initial random trials and 38 sequential trials selected by expected improvement from 2,048 random candidate locations per trial. It uses a Gaussian process with a Matern 5/2 covariance, fixed length scale and a fixed noise term. Exploration ranges from 0 to 0.50. Learning rate ranges from 0.01 to 1.0 on a logarithmic scale. The fixed noise models the uneven score surface, including changes to random action sequences. These GP assumptions are design choices, not guarantees that its uncertainty estimates are calibrated. Each trial averages 30 development markets (seeds 700000–700029). Settings are fixed before comparison on 100 new seeds (800000–800099). A fresh agent is created for each 200-pull run.

The complete GP trial history is in `bandit_gp_trials.csv`. Exact selected parameters, individual rewards for the fresh seeds and official seeds, and comparisons are in `bandit_gp_evaluation.json`.

| Agent and settings | 100 new seeds | 5 official seeds |
|---|---:|---:|
| GP-selected epsilon-greedy | 1.24581 | 1.12631 |
| Prior wider-grid epsilon-greedy (0.02, 0.1) | **1.29698** | **1.31772** |
| Original epsilon-greedy (0.15, 0.2) | 1.22248 | 1.10774 |
| Sliding-window UCB | 1.10342 | 0.97594 |
| Random baseline | 0.22948 | 0.33750 |
| Reference epsilon-greedy | 0.45504 | 0.20705 |

The prior wider-grid settings scored highest on the new 100-seed comparison and on the official five seeds. They are now the defaults in `bandit_agent.py`. Selecting between methods after viewing the new comparison makes that comparison part of the selection process. Another independent set would be needed for an untouched estimate of the finally selected agent. These tests use one fixed agent-randomness setup. Also, the random strategy consumes simulator randomness, so the same market seed does not always yield identical paths for agents that make different choices. A bounded 48-trial search does not prove a global optimum.

Phase 1 has no discount factor: the bandit receives one reward for each strategy choice and does not estimate future state values. Exploration and learning rate are the two parameters of this epsilon-greedy implementation.

## Final confirmation

Run `python -B verify_bandit_selection.py` to reproduce the wider grid and the confirmation on 300 previously unused seeds, 900000–900299. All five configurations were fixed before this run. The full scores by seed are in [bandit_final_check.csv](bandit_final_check.csv). The wider-grid choice also reproduced its official local score of 1.31772.

| Configuration | Mean score | Wider-grid lead (95% paired bootstrap interval) |
|---|---:|---:|
| Wider-grid epsilon-greedy | **1.24881** | — |
| GP-selected epsilon-greedy | 1.20811 | +0.04071 (−0.04605 to +0.12787) |
| Sliding-window UCB | 1.09758 | +0.15123 (+0.07343 to +0.22982) |
| Original epsilon-greedy | 1.06257 | +0.18625 (+0.10029 to +0.27269) |
| Discounted Thompson sampling | 1.00423 | +0.24458 (+0.16534 to +0.32219) |

The paired intervals resample market seed IDs together across methods. The wider-grid choice has the highest observed average, but the interval against the GP candidate includes zero. The selected values are the best found in the wider grid, and their advantage is conditional on the methods, settings, simulator, and agent randomness tested here.

For a future experiment, first try a denser grid near 0–5% exploration and learning rates around 0.1. The wider-grid winner was on the exploration boundary, and the GP candidate left its relative performance uncertain. A separate experiment could let exploration rise after a sustained drop in recent rewards, or retune the UCB window and exploration bonus to adapt more quickly. Select any new settings on development seeds and reserve a different set for confirmation.
