# Phase 2: reinforcement learning with a KL leash to the imitation policy

## Goal

Make the imitation Fox (`runs/bc_fox/bc_fox_18h.pt`) stronger while it still plays like a
person. The owner's priority, in their words, is that it "should still feel like human":
a version that wins more but looks robotic is worse than one that is a little stronger and
fully human-like. Problems seen in their games that RL should fix: games lost without
combos, and self-destructs offstage (failed recoveries from the side). Budget: 2 nights.

## Approach

PPO from the imitation policy, with a per-frame penalty `kl_coef × KL(policy ‖ teacher)`
where the teacher is the frozen imitation policy (slippi-ai's recipe). Stakes decide how
far it moves: a frame that decides a stock (recover or die) has a large advantage and can
shift; everyday habits have small advantages and stay human.

- **Opponents** (10 games): 6 self-play, 2 Ganondorf bot (a friend's SB3 agent, no reaction
  delay, decides every 3 frames), 2 level 9 CPU Fox. Self-play opponents: the current
  policy (50%) or a random earlier snapshot (snapshots every 20 min; the imitation policy
  is in the pool), re-picked every 5 minutes. The opponent Fox also has the 18-frame
  delay and sees the game from its side.
- **Stage and delay:** Final Destination only; 18 frames for both Foxes.
- **Reward:** ±1 per stock, ±0.01 per percent; -0.02 per ledge grab (planking), -0.001 per
  frame offstage after 3 s (stalling), +0.0002 per unit moved towards the opponent
  (running away). The shaping terms are small next to a stock (a full approach across FD
  ≈ 0.02).
- **PPO:** rollouts of 512 frames per game; minibatches of 128-frame chunks that start from
  the LSTM state stored while acting (first minibatch reproduces acting exactly); games
  that end mid-chunk restart the LSTMs via a (B, T) reset mask. γ = 0.997 per frame
  (half-life ~4 s, a little over the 18-frame delay plus a punish), λ = 0.98, clip 0.2,
  Adam 3e-5 (policy) and 1e-4 (value), 2 epochs × 4 minibatches, KL coefficient 0.5,
  no entropy bonus (the KL keeps the policy's spread).
- **Value network:** a copy of the imitation core with a zero-initialised head, trained
  alone for the first 30 updates so early advantages aren't noise.
- **Throughput:** acting is one CUDA graph replay per frame (0.8 ms vs 3.7 ms eager).

## Monitoring and success

- KL to the teacher per frame (the "human-likeness budget"); if it keeps climbing past
  ~0.1, raise `--kl-coef`.
- Win rates over the last 50 games per group (self-play, Ganondorf, CPU 9).
- After training: `evaluate.py` on snapshots vs CPU 9 and the Ganondorf bot, the Phase 1
  style metrics (combo length, move variety, wavedashes, L-cancels) against the human
  numbers, and the owner's own games with `play.py`.

## Robustness for unattended nights

A crashed or frozen Dolphin ends only its game (truncated); the next reset starts a new
Dolphin. `last.pt` every 10 minutes and on Ctrl+C; `--resume` continues.
