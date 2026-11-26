# RL Trading Arena — Student Starter Kit

A single-player trading environment with a class leaderboard, run as a
two-phase semester project: **multi-armed bandits → tabular Q-learning / SARSA**.

Read `RL_Trading_Arena_Project_Spec.docx` first — it is the full assignment
brief. This file is just the quick reference for the code.

Nothing to install beyond `numpy`.

---

## 1. What's in the kit

| File | Purpose |
|---|---|
| `bandit_env.py` | Phase 1 environment: non-stationary 5-armed bandit over trading strategies |
| `trading_env.py` | Phase 2 environment: regime-switching market as a 30-state / 3-action tabular MDP |
| `agent_base.py` | The interfaces you implement (`BanditAgent`, `MDPAgent`) |
| `baseline_agents.py` | Reference agents — the leaderboard "floor" you should beat |
| `eval_harness.py` | The fixed-seed scoring logic (the same code the server runs) |
| `student_template.py` | Blank scaffold — copy this to `agent.py` and fill in the TODOs |
| `test_locally.py` | Score your own agent without touching the server |
| `submit_agent.py` | Submit your agent to the leaderboard |

---

## 2. Setup

Python 3.10 or later, plus numpy:

```bash
python3 --version
pip install numpy --break-system-packages   # if you don't already have it
```

Confirm everything runs before you write any code:

```bash
python3 trading_env.py      # prints a portfolio value and the state/action counts
python3 bandit_env.py       # prints cumulative reward per strategy
```

---

## 3. Your workflow

1. Copy `student_template.py` → `agent.py`, and fill in the TODOs for
   whichever phase is currently active. Your file must define a class named
   exactly `Agent`, and only one of them.
2. Iterate locally — this runs the *exact* scoring logic the server uses, so
   you don't have to wait on submissions:
   ```bash
   python3 test_locally.py agent.py bandit
   python3 test_locally.py agent.py mdp
   ```
3. Submit when you're happy with it:
   ```bash
   python3 submit_agent.py agent.py <your_student_id> bandit
   python3 submit_agent.py agent.py <your_student_id> mdp <server_url>
   ```
4. Check the standings at `<server_url>/leaderboard.html?phase=bandit` (or
   `?phase=mdp`).

Your instructor will give you the server URL and the resubmission and
deadline rules.

---

## 4. How you're scored

- **Phase 1 (bandit):** average cumulative reward over 5 independent 200-pull
  replicate runs on 5 fixed seeds. You get a fresh agent instance per replicate.
  The market walks through three phases during each run — bull-heavy, then
  range-bound, then bear-heavy — and **each phase has a different best arm**.
  Picking a good arm is easy; noticing when it stops being good is the
  assignment.
- **Phase 2 (MDP):** you train for 300 episodes on a fixed set of seeds, then
  `set_eval_mode(True)` is called and you are scored on 50 **held-out** episodes
  you never trained on. Score is risk-adjusted:
  `score = mean_return - 0.5 * mean_max_drawdown`.

Section 3 of the project spec explains why it's built this way — read it before
you start tuning.

---

## 5. Two things to know going in

**Phase 1.** The bundled `EpsilonGreedyBanditAgent` (fixed epsilon = 0.1, no
decay) scores *below* the random baseline. That is not a typo and not a
broken baseline: under a market that changes underneath it, a learner that
assumes the arms are stationary commits to a stale winner, while random at
least keeps sampling everything. Hardcoding a single arm does not rescue you
either — the best fixed arm still loses to an agent that tracks the switches.
Explaining that gap is a good first exercise.

**Phase 2.** The bundled `QLearningReferenceAgent` (fixed epsilon = 0.1,
fixed alpha, no decay) scores just above 0 — it trades timidly and leaves
most of the available edge on the table. Also deliberate. Working out *why*
naive tabular Q-learning under-explores a small, noisy reward signal — and
what fixes it (a longer discount horizon, a higher starting epsilon with a
decay schedule, more training episodes, SARSA vs Q-learning) — is a large
part of the assignment, and of the final report.

One knob that will **not** help, and is worth understanding rather than
just trying: multiplying your reward by a constant before the Q-update.
Scaling every reward scales every Q-value by the same factor, and `argmax`
does not care — your policy, and your score, come out identical.
