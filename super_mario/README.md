# Super Mario Bros PPO Agent

Trains a PPO agent (Stable-Baselines3) to clear World 1-1 of the original NES
Super Mario Bros, using the `gym-super-mario-bros` emulator.

## Setup

```bash
./setup.sh                    # creates .venv with Python 3.10 and installs everything
source .venv/bin/activate
python test_env.py            # validates the env, runs a random agent, saves sample frames
```

The emulator needs the old `gym` 0.26 and `numpy < 2`; these are pinned in
`requirements.txt`. The "Gym has been unmaintained" message printed on startup is
harmless.

## Train

```bash
python train.py                                   # World 1-1, 10M steps, 8 parallel games
python train.py --timesteps 200000 --run-name quick   # short trial run
python train.py --save-every 5                    # save weights every 5 PPO updates (default 10)
python train.py --n-envs 10                       # ~15% faster on an M4, but uses all cores
python train.py --resume runs/<run>/checkpoints/ppo_mario_update0100_409600steps.zip
```

Press Ctrl+C at any time: the current model is saved to `runs/<run>/final_model.zip`.
For a run started in the background, `pkill -f "run-name <run>"` does the same.

**Checkpoints.** Weights are saved every `--save-every` PPO updates to
`runs/<run>/checkpoints/ppo_mario_update<N>_<steps>steps.zip`. One update is one
rollout (8 envs x 512 steps = 4,096 game steps) followed by 4 optimization
epochs over it; it takes about 11 seconds on an M4. Each checkpoint is about
21 MB because it includes the optimizer state needed for `--resume`, so a full
10M-step run uses about 5 GB with `--save-every 10` or 10 GB with `--save-every 5`.

On an Apple M4 training runs at about 380 steps/s (8 envs, Apple GPU via MPS),
so 1M steps take about 45 minutes and the full 10M run about 7-8 hours. The
agent usually starts reaching the flag somewhere in the 2-6M step range, but this
varies between seeds.

## Monitor

```bash
tensorboard --logdir runs        # open http://localhost:6006
```

Most useful curves:

| Tag | Meaning |
|---|---|
| `mario/max_x_pos` | How far Mario gets on average (flag is at about 3160) |
| `mario/clear_rate` | Fraction of episodes that reach the flag |
| `rollout/ep_rew_mean` | Average episode reward |
| `train/entropy_loss` | Exploration; slowly approaches 0 as the policy becomes confident |

## Watch the agent

```bash
python evaluate.py runs/<run>/final_model.zip              # opens a game window (q to quit)
python evaluate.py runs/<run>/final_model.zip --episodes 10 --no-render --stochastic
python evaluate.py runs/<run>/checkpoints --episodes 5 --stochastic   # score every checkpoint
```

Passing a folder evaluates every checkpoint in it without a window, prints a
table of clears and distance per checkpoint, names the best one, and saves the
table to `eval_results.csv` in that folder.

The game is deterministic, so with the default (deterministic) actions every
episode plays out identically. Use `--stochastic` to measure a real clear rate.

## How it works

- **Actions:** 7 button combos (`SIMPLE_MOVEMENT`: noop, right, right+A, right+B, right+A+B, A, left).
- **Observations:** each action is repeated for 4 frames; frames are max-pooled
  (removes sprite flicker), converted to grayscale, resized to 84x84 and the last
  4 are stacked so the network can see motion.
- **Reward:** the built-in reward (moving right, minus time and a death penalty)
  plus a +50 bonus for the flag, all scaled by 0.1.
- **Episode end:** reaching the flag, dying (the emulator ends single-level
  episodes on the first death), or 300 steps without moving further right.
- **PPO:** CNN policy, 512-step rollouts x 8 envs, batch 256, 4 epochs,
  learning rate 2.5e-4 decaying linearly to 0, gamma 0.99, entropy coefficient 0.01.
  All of these are command-line flags; run `python train.py --help`.

## Files

| File | Purpose |
|---|---|
| `mario_rl/env.py` | Gymnasium adapter and preprocessing for the emulator |
| `train.py` | PPO training with per-update checkpoints and TensorBoard stats |
| `evaluate.py` | Watch a model play, or score a whole folder of checkpoints |
| `test_env.py` | Environment sanity check and speed test |
