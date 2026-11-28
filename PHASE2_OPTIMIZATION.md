# Phase 2 hyperparameter optimization

Both tuned agents improved on their previous official local scores. SARSA had the highest average on the 100 fresh episodes and remains the recommended Phase 2 agent. Its advantage over tuned Q-learning was small relative to the variation between episodes.

## What happened under the hood

Both agents start with a table of zeroes. Each row is one market state and each column is one action: stay flat, hold, or go long. The simulator returns a reward after each trading action, including any transaction cost. The agent updates the table entry for the action it took.

- **Q-learning** updates toward the immediate reward plus the discounted value of the *highest-valued* next action. It can learn about a greedy policy while deliberately exploring during training.
- **SARSA** updates toward the immediate reward plus the discounted value of the next action it *actually picks*. The picked action can be exploratory, so the update reflects the behavior of the training policy. The agent saves that choice and takes it on the next turn.
- Both methods use only the immediate reward on the last day. In evaluation, both pick the highest-valued action in their table and stop learning. The evaluator scores the resulting portfolio paths separately from the daily reward used for training.

For example, suppose today's reward is `0.01`, the highest estimated value in the next state is `0.03`, and SARSA actually picks an action worth `0.01`. At `gamma = 0.95`, Q-learning moves toward `0.01 + 0.95 × 0.03 = 0.0385`, while SARSA moves toward `0.01 + 0.95 × 0.01 = 0.0195`. The learning rate determines how far the old estimate moves toward that target.

## How the search worked

We used **exhaustive grid search** over a fixed set of hyperparameters. The grid is small enough to cover every combination, and both algorithms get the same search budget. This finds the best development score within the specified grid. A different grid, a different training run, or more evaluation episodes could change which settings win. [Bergstra and Bengio, *Random Search for Hyper-Parameter Optimization*](https://jmlr.org/papers/v13/bergstra12a.html)

| Setting | Values tested | What it changes |
|---|---|---|
| Learning rate (`alpha`) | `0.02`, `0.05`, `0.10` | How much each update changes a table entry |
| Discount factor (`gamma`) | `0.90`, `0.95`, `0.99` | How much weight future rewards receive |
| Starting exploration | `0.10`, `0.30`, `0.50` | Chance of choosing randomly early in training |
| Ending exploration | `0.01`, `0.05`, `0.10` | Chance of choosing randomly by the end of training |
| Exploration schedule | Linear, quadratic | Quadratic reduces exploration faster at first and both reach the same endpoint |

The script scored 162 combinations per algorithm, reusing nine results per algorithm where equal endpoints made both schedules constant. Each distinct configuration trained on the assignment's 300 episodes (seeds `0–299`). The script ranked configurations on 20 development episodes (seeds `200000–200019`) by **average return minus half of average maximum drawdown**. Drawdown is the largest percentage drop from a prior portfolio peak within an episode.

After choosing the settings on development episodes, the script evaluated them on 100 fresh holdout episodes (seeds `300000–300099`) and the 50 official local episodes (seeds `100000–100049`). The fresh episodes assess performance after selection. The official episodes show what the starter scoring setup gives and were also used in earlier work. [Cawley and Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*](https://www.jmlr.org/papers/v11/cawley10a.html)

The script uses the same market seed for each agent in each evaluation episode. It estimates 95% intervals for score differences by resampling paired episodes 10,000 times. These intervals describe uncertainty from evaluation markets for the trained agents. The experiment uses one fixed training setup, so the intervals do not capture how much the results would vary after retraining with different training or exploration seeds.

## Results and what they indicate

Both algorithms selected a learning rate of `0.02`, a discount of `0.99`, and starting exploration of `0.50`. Q-learning selected **linear decay to `0.10`** and scored `0.02234` on development episodes. SARSA selected **quadratic decay to `0.05`** and scored `0.01952` on development episodes. These are now the default settings in the two agent files.

| Agent | Fresh score (100) | Official local score (50) | Fresh return | Fresh max drawdown |
|---|---:|---:|---:|---:|
| Provided Q-learning baseline | 0.02076 | 0.02788 | 3.77% | 3.39% |
| Previous Q-learning | 0.01303 | 0.02159 | 2.80% | 3.00% |
| Tuned Q-learning | 0.01948 | 0.03341 | 3.73% | 3.57% |
| Previous SARSA | 0.01942 | 0.02985 | 3.61% | 3.34% |
| **Tuned SARSA** | **0.02125** | **0.03626** | **4.01%** | 3.76% |

On fresh markets, tuned Q-learning improved by `0.00645` over its previous version (95% interval `−0.00004` to `+0.01336`). Tuned SARSA improved by `0.00183` (95% interval `−0.00393` to `+0.00769`). Both intervals include zero. Tuned SARSA led tuned Q-learning by `0.00177` (95% interval `−0.00494` to `+0.00858`) and beat the provided baseline by just `0.00049` on average. Those results support recommending SARSA based on observed scores, with uncertainty about the size and direction of its advantage on other markets.

### How their trading changed

- **Q-learning** went from being invested on `20.2%` of fresh evaluation days to `26.8%`, while position changes fell from `5.46` to `4.64` per episode. The higher average return outweighed the larger drawdown penalty compared with its previous version. Its fresh score remained slightly below the provided baseline.
- **SARSA** went from being invested on `24.5%` of days to `28.8%`, while position changes fell from `4.74` to `4.26` per episode. It also accepted more drawdown, but the extra return was enough to raise its average score.
- **With matching settings**, Q-learning was ahead on fresh episodes when both used Q-learning's selected schedule (`0.01948` versus `0.01648`). SARSA led when both used SARSA's selected schedule (`0.02125` versus `0.01626`). Both paired intervals include zero. Q-learning using SARSA's settings scored `0.04409` on the official local episodes, but on fresh episodes it stayed invested `45.9%` of the time, with `5.92%` mean maximum drawdown. The higher fresh return came with enough extra drawdown to leave it behind SARSA on the fresh score. This illustrates why the highest observed official score alone did not determine the selected defaults.

## Reproduce the experiment

```powershell
python -B optimize_phase2.py
python -B test_locally.py q_learning_agent.py mdp
python -B test_locally.py sarsa_agent.py mdp
```

The search writes [all grid scores](phase2_grid_results.csv) and the [evaluation summary](phase2_evaluation.json). The evaluation summary includes the exact settings, return, drawdown, score, invested days, and position changes. All evaluations ran locally.
