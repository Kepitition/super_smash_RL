# Phase 1: imitation-learned Fox bot (design)

Part of a three-phase plan to train a Melee agent that plays with real combos
instead of spamming:

1. **Phase 1 (this spec):** learn Fox from human replays (behavior cloning), with
   controls rich enough for real Melee technique.
2. Phase 2: PPO self-play from the Phase 1 policy, kept close to it with a KL
   penalty (~4 nights of training).
3. Phase 3: evaluation (combo and move-variety metrics, checkpoint ladder) and a
   Windows play script for a human.

Research basis: slippi-ai (vladfi1, MIT) and Faynt (arXiv 2610.02144) both reach
pro-level, human-looking play with exactly this recipe: imitation on Slippi replays,
then RL with a KL leash. Phase 1 settings follow slippi-ai's published Fox
imitation config unless noted.

## Decisions

| Topic | Decision |
|---|---|
| Character | Fox (one model; character stays an input so others can be added later) |
| Data | Fox games, rank pairs master-master + master-diamond, from `erickfm/melee-ranked-replays` (MIT): ~84k games, ~93 GB compressed |
| Stages | Imitation on the 6 legal stages; Phase 2 RL starts on Final Destination |
| Reaction delay | 18 frames: the controller chosen at frame t is applied at frame t+18 |
| Decision rate | Every frame (60 Hz). Fallback if live throughput is too low: every 2 frames |
| Model | Recurrent "transformer-like" policy (LSTM + feed-forward blocks), ~10M parameters |
| Compute | Overnight runs (8-12 h) that checkpoint and resume; imitation over 1-2 nights |
| Storage | `D:\smash_data` (configurable), ~50 GB parsed |
| Parallel games | Tuned by measurement between 10 and 20 (mainly matters in Phase 2) |

## 1. Data pipeline (`prepare_data.py`)

For each archive `FOX/FOX_{master-master,master-diamond}_a{1..6}.tar.gz`:

1. Download to `D:\smash_data\raw\`, skipping archives already marked done.
2. Read the tarball member by member (no full extraction) and parse each `.slp`
   with peppi-py in a process pool (all cores).
3. Keep a game only if all hold:
   - exactly 2 human players, at least one of them Fox;
   - stage is one of Final Destination, Battlefield, Dream Land, Yoshi's Story,
     Fountain of Dreams, Pokemon Stadium;
   - at least 1800 frames (30 s) of play;
   - the game ended normally (stocks or time), not a quit-out or disconnect.
4. Write one compressed file per game, `D:\smash_data\parsed\<archive>\<game_id>.npz`,
   holding the raw frames (section 2) for both players.
5. Append the game to `D:\smash_data\index.jsonl` (game id, archive, rank pair,
   stage, both characters, Fox ports, frame count, split), then delete the archive
   once all its games are written.

Split: 2% of games, chosen by a hash of the game id, go to `val`, the rest to `train`.
The script resumes from the index after interruption. Expected cost: ~30 min of
downloading, 1-2 h of parsing, ~50 GB on disk.

## 2. Shared raw frame format and codec

The key consistency rule: replays and live games are both converted to the same
**raw frame** record, and everything the model sees is computed from raw frames by
one codec. Nothing model-specific is stored on disk, so changing the features never
requires re-parsing.

### Raw frame (`smash_rl/state.py`)

Per frame: `stage` and, per player:

- position x, y; percent; stocks; facing; on ground; invulnerable;
- hitlag frames left; hitstun frames left; jumps left; shield strength;
- action state id; action frame; character id;
- self-induced speeds (air x, ground x, y) and knockback speeds (x, y);
- controller: main stick x, y and C-stick x, y (processed, -1..1), analog trigger
  (0..1, max of L and R), and the digital buttons A, B, X, Y, Z, L, R.

Two producers fill this record: `from_peppi(game)` for replays and
`from_libmelee(gamestate)` for live play.

### Model inputs (`smash_rl/codec.py`)

Built from the agent's and the opponent's raw frames, always agent first:

- **Numbers:** x, y (×0.05), percent (×0.01), stocks (/4), facing (±1), on ground,
  invulnerable, hitlag and hitstun (/30, capped at 4), action frame (/60, capped at 2),
  shield (×0.01), the 5 speeds (×0.5).
- **Jumps left:** one-hot of 7.
- **Learned embeddings:** action state (400 ids → 32), character (33 → 16),
  stage (7 = six legal stages + other → 8).
- **The agent's previous controller** (its own last decision), encoded as in the
  next subsection.

Not in Phase 1: projectiles, moving-platform heights. Both are additions to the
raw frame and codec if evaluation shows they are needed.

### Controller (the action)

Chosen in four autoregressive steps, each conditioned on the earlier ones:

1. **Buttons (48 classes):** A × B × jump (X or Y) × Z × trigger
   {none, light (0.35 analog), full (digital L + analog 1.0)}.
2. **Main stick x (17 classes):** -1 to 1 in steps of 0.125.
3. **Main stick y (17 classes):** same grid.
4. **C-stick (5 classes):** neutral, up, down, left, right.

From replays, each frame's controller is snapped to the nearest class: X and Y
merge into jump; sticks snap per axis to the nearest 0.125 step; the trigger is
full if digital L or R is pressed, light if the larger analog value is at least
0.3 (Melee's light-shield threshold, 43/140), and none otherwise. Start and the
D-pad are never pressed.

To live play, a class tuple is decoded to stick values, the analog trigger and
buttons, and sent through the buffered libmelee controller already in `env.py`.

### Consistency tests (both must pass before training)

- **Round trip.** In a live game, send every stick class, trigger level and button
  combination through the pipe, read the game's own recorded controller back from
  the next frames, and check it decodes to the same class.
- **Replay vs live.** Play a short live game with random controller classes and
  Slippi replay saving on. Convert the game two ways: live (`from_libmelee` each
  frame) and the saved `.slp` (`from_peppi`). Check the raw frames match frame by
  frame, field by field, within float tolerance.

## 3. Env changes (`smash_rl/env.py`, `smash_rl/dolphin.py`)

- **Split the module.** Today's Windows Dolphin layer (per-port copies, buffered
  controller, robust stop and connect, job object, menus, lean workers) moves
  unchanged into `smash_rl/dolphin.py`; `env.py` keeps the Gym environment.
- **Per-frame steps.** One step is one frame. The action is a controller class
  tuple; the observation is the raw frame for both players plus the stage.
  `frame_skip`, `ACTIONS` and the 898-float observation are removed.
- **Delay queue.** The env keeps an 18-entry queue per controlled port. Each step
  pushes the new decision and applies the oldest one. The first 18 frames of a game
  apply neutral input. The delay is an env argument, default 18.
- **Stage per reset.** `reset(options={"stage": ...})` or a fixed stage list, chosen
  through the existing menu helper.
- **Opponent.** A CPU (character, level), or `HUMAN` (port 2 is a Dolphin-mapped
  controller). The `AGENT` mode for self-play is Phase 2.
- **Reward.** Stays the 0 placeholder (Phase 2).
- **Replays.** An optional `save_replays` directory makes Dolphin write `.slp` files
  (used by the replay-vs-live test and for watching games).

## 4. Model (`smash_rl/model.py`)

- **Encoder.** Concatenate the model inputs (section 2) → linear → 512.
- **Core.** 3 blocks of [LayerNorm → LSTM(512), residual] + [LayerNorm →
  FFW 512→1024→512, residual]. This is slippi-ai's `tx_like` (3 layers, 512 wide,
  ffw ×2), about 9.5M parameters. The recurrent state is carried through a whole
  game.
- **Controller head.** Autoregressive. Each of the 4 steps is a small MLP (128
  hidden) over the core output plus embeddings of the classes already chosen.
  Training uses teacher forcing (the human's choices); acting samples
  (temperature 1).
- **Value head.** None in Phase 1. Phase 2 adds a separate value network and may
  pretrain it on the same replays.

## 5. Imitation training (`train_bc.py`)

- **Samples.** Each Fox in a train game is one trajectory (a Fox ditto gives two).
  The input at frame t uses raw frames up to t; the label is that Fox's controller
  at frame t+18. The previous-controller input is the label for t+17.
- **Batches.** Stateful unrolls: 512 parallel readers each walk through a
  trajectory in chunks of 80 frames, carrying the LSTM state between chunks. A
  reader that finishes its game takes a random new one with a zero state. This
  matches how the bot runs live.
- **Loss.** Sum of the 4 cross-entropies.
- **Optimizer.** AdamW, learning rate 1e-4, gradient clipping at 1.0, bf16 autocast
  on the RTX 4060. If memory runs out, lower the batch size, then the width.
- **Loading.** Worker processes load the per-game `.npz` files and compute codec
  features vectorised per chunk.
- **Runtime.** Every 10 minutes: save a resumable checkpoint (model, optimizer,
  step) and compute loss and per-head accuracy on a fixed set of val trajectories;
  keep `best.pt` by val loss. TensorBoard logs. `--max-hours` stops cleanly
  overnight; `--resume` continues.
- **Budget.** 1-2 nights; stop when val loss has not improved for ~2 h.

## 6. Phase 1 evaluation and success criteria

`evaluate.py`: the bot (port 1) plays N games against a CPU (character, level,
stage) at full speed or watchable speed, saves `.slp` replays, and prints win rate,
mean stock difference and damage per minute. The bot runs in the main process: one
batched GPU forward per frame across all games, with the LSTM state carried per game.

Phase 1 is done when:

1. both consistency tests pass;
2. imitation training finished, with val loss and per-head accuracy reported;
3. against CPU level 9 Fox on Final Destination over 20 games: it completes every
   game without getting stuck, and the win rate is reported (target ≥ 70%;
   Phase 2 RL is what pushes strength);
4. watching saved replays shows human Fox technique: short hops, L-cancels,
   wavedashes or waveshines, and combos longer than one hit.

## 7. Testing

- **pytest unit tests:** controller class encode/decode round trips; the
  replay-label snapping rules; delay alignment (label at t+18, previous controller
  at t+17); codec shapes and scales; the replay filters; the index/resume logic.
- **Live tests (scripts, need Dolphin):** the two consistency tests; `test_env.py`
  updated for per-frame steps, including a speed measurement with N games for
  picking the parallel-game count.
- **Before overnight runs:** a 2-minute smoke run of `train_bc.py` on ~20 games
  (loss goes down, checkpoint and resume work).

## 8. Files

| Path | Purpose |
|---|---|
| `smash_rl/dolphin.py` | Dolphin process and Windows layer (moved from `env.py`) |
| `smash_rl/state.py` | Raw frame record; `from_peppi`, `from_libmelee` |
| `smash_rl/codec.py` | Model inputs from raw frames; controller classes, snapping, decoding |
| `smash_rl/env.py` | Per-frame Gym env with delay queue, stage per reset, CPU/HUMAN opponent |
| `smash_rl/model.py` | Recurrent policy with autoregressive controller head |
| `smash_rl/data.py` | Index, splits, stateful unroll batches |
| `prepare_data.py` | Download, filter, parse, index |
| `train_bc.py` | Imitation training |
| `evaluate.py` | Bot vs CPU, replays, win rate |
| `test_env.py` | Updated sanity and speed check |
| `tests/` | pytest unit tests and the consistency scripts |

New dependencies: `peppi-py`, `huggingface_hub`, `pytest` (dev).

## 9. Risks

- **Per-frame throughput.** Lockstep stepping of 10-20 games one frame at a time
  may be slow. Measure first; the fallbacks are fewer games or a decision every
  2 frames (the delay then counts in 2-frame units).
- **Replay vs live mismatch** (field definitions, libmelee "fixes" such as
  zero-indexed action frames). Caught by the replay-vs-live test before any
  training.
- **Compounding errors.** Imitation bots can drift into states humans never
  visit (e.g. loops at the ledge). Watch replays in evaluation; Phase 2 RL
  corrects this.
- **GPU memory (8 GB).** Batch 512 × 80 frames at 512 wide should fit with bf16;
  otherwise lower the batch.
- **Download limits** on Hugging Face. The script resumes per archive.

## Out of scope for Phase 1

Rewards, PPO, self-play (`AGENT` opponent), value network, combo and move-variety
metrics, checkpoint ladder, the Windows play script, projectiles, other characters,
moving-platform heights.
