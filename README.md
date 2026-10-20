# Super Smash Bros. Melee RL

A Fox bot for Super Smash Bros. Melee (NTSC v1.02) that learns from human replays
first (imitation), then improves by reinforcement learning, so it plays with real
technique and combos instead of spamming one move. It plays every frame with a full
GameCube controller and a human-like 18-frame reaction delay, through the Slippi
Dolphin emulator and [libmelee](https://github.com/vladfi1/libmelee), on Windows (Linux
and macOS still work).

This started from [KungAlpaca/DML_RL_SmashBros](https://github.com/KungAlpaca/DML_RL_SmashBros):
only its Dolphin environment was kept (the original reward, training, evaluation and
play scripts are in the git history).

**Status:** Phase 1 (imitation learning) is done; Phase 2 (PPO self-play with a KL
penalty towards the imitation policy, `train_rl.py`) is ready to train. Evaluation and
a play script exist (`evaluate.py`, `play.py`).
Design: `docs/superpowers/specs/`, plans: `docs/superpowers/plans/`.

## Setup (Windows)

1. Put your own Melee NTSC v1.02 ISO in this folder as `Super Smash Bros Melee v1.02.iso`
   (game ID `GALE01`, revision 2). It is never committed.
2. `powershell -ExecutionPolicy Bypass -File setup.ps1` creates `.venv` with Python
   3.12 / 3.11 / 3.10 (numpy<2 has no wheels for 3.13+), installs PyTorch (CUDA build
   if an NVIDIA GPU is found; `-Cpu` for the CPU build) and the requirements, and
   downloads the portable Slippi Dolphin 3.6.4 into `dolphin\`.
3. Once, so Windows stops asking to allow every new Dolphin on networks (asks for admin):
   `powershell -ExecutionPolicy Bypass -File windows_firewall.ps1` (ports 51441-51700;
   `-BasePort` / `-Count` to change). Everything talks to Dolphin over localhost.
4. Check it: `.venv\Scripts\Activate.ps1`, then `python test_env.py` and `python -m pytest`.

Linux / macOS: `./setup.sh`, then `source .venv/bin/activate`. Only Linux can run
headless (the ExiAI Dolphin build is a Linux AppImage).

## The environment (`smash_rl/env.py`)

- **One step = one frame.** The action is one controller, as four classes
  (`smash_rl/codec.py`): buttons (A, B, jump, Z, trigger none / light / full: 48
  classes), main stick x and y (17 positions each, -1..1), C-stick (neutral, up, down,
  left, right). That covers short and full hops, wavedashes, L-cancels and C-stick
  aerials.
- **Reaction delay.** `delay` (default 18): a decision made after seeing frame t is the
  controller of frame t+18, through a queue in the env.
- **Observation:** raw frames (`smash_rl/state.py`, `PLAYER_DTYPE`) for
  `[agent, opponent]`: position, speeds, percent, stocks, facing, on ground,
  invulnerable, hitlag, hitstun, jumps, shield, action state and frame, character and
  the controller; plus the stage index and frame number. `codec.encode_state` turns
  them into model inputs. Replays (peppi-py) and live games (libmelee) produce exactly
  the same records (checked frame by frame by `tests/live/check_replay_vs_live.py`).
- **Episode:** one full game, from its first frame (-123, the countdown). Port 2 is a
  CPU (`opponent`, `cpu_level`), a random CPU character every game (`RANDOM`), a
  person (`HUMAN`), a bot (`opponent_bot`, e.g. the Ganondorf bot below) or a second
  policy (`POLICY`, self-play: each step then takes two controllers, both delayed).
  `stages`: one or more stage names. `save_replays`: a directory for Slippi replays
  (one `port<port>` subfolder per game, since Slippi names replays by the second).
  `reward`: weights for `smash_rl/reward.py` (else the reward is 0). A Dolphin that
  crashes or freezes mid-game ends only that game; the next reset starts a new Dolphin.
- **Parallel games:** `smash_rl.vec_env.make_vec_env(n_envs, base_port, ...)`, one
  worker process (~50 MB, no PyTorch) and one Dolphin per game, stepped in lockstep.

`test_env.py` checks inputs arrive and measures speed: `--n-envs --frames --cpu-level
--opponent --stage --delay --port --headless`.

### Speed (i9-14900HX, 32 GB, RTX 4060; random controllers, one decision per frame)

| Games | Frames/s, total | Frames/s, per game |
|---|---|---|
| 1 | 366 | 366 |
| 10 | 1238 | 124 |
| 20 | 1209 | 60 |

The total stops growing around 10 games, which is the sweet spot (each Dolphin needs
~610 MB of RAM). Run `python test_env.py --n-envs N --port 51600` to compare.

### What Windows needed (`smash_rl/dolphin.py`)

- **Per-port Dolphin copies.** Windows Dolphin names its input pipes
  `\\.\pipe\slippibot1..4` for every instance, so parallel Dolphins collided. Each Slippi
  port gets its own copy in `dolphin\instances\slip<port>\` (hard links, except a
  patched exe), made on first use.
- **Truly blocking input.** Dolphin 3.6.4's Windows build returns from its input loop
  when the bot's input hasn't arrived yet, so frames ran on stale input and the reaction
  delay drifted by up to ~90 frames with several games. The copies are patched to wait
  for the input, as mainline Slippi Dolphin does (`patch_dolphin_exe`). Measured lag
  is now 0 frames.
- **Inputs sent whole**, once per frame, as on Linux (libmelee wrote each command
  separately on Windows).
- **Clean shutdown and startup.** Dolphin is killed before libmelee's helper; all
  workers and Dolphins die with the main process (Windows job object); slow parallel
  starts wait up to `connect_timeout` (120 s).
- Saving replays turns off instant restarts: Slippi only finishes a replay file when
  the game ends normally.

## Data (`prepare_data.py`)

Fox games from [erickfm/melee-ranked-replays](https://huggingface.co/datasets/erickfm/melee-ranked-replays)
(MIT), rank pairs master-master and master-diamond: 111,310 kept games (1.24 billion
Fox frames), ~93 GB of archives, 35 GB once parsed, about 1 hour in total. One archive
at a time is downloaded, parsed on all cores and deleted. Kept games: 2 humans, one of
the 6 legal stages, at least 30 s of play, and finished (a player out of stocks or the
timer ran out; the dumps have no end record). Online replays repeat frames after
rollbacks; only the last copy of each frame is kept.

```
python prepare_data.py                       # everything, into D:\smash_data (resumes if stopped)
python prepare_data.py --ranks master-master --archives 1 2
```

Flags: `--data-dir --character --ranks --archives --workers --keep-archives --max-games`.
Layout: `parsed/<archive>/<game>.npz` (raw frames of both players), `index.jsonl`
(one line per game: stage, characters, Fox slots, frames, train/val split), `done/`
(finished archives). 2% of games are validation.

## Imitation training (`train_bc.py`)

A recurrent policy (`smash_rl/model.py`, ~9.9M parameters: 3 blocks of LSTM +
feed-forward, 512 wide, autoregressive controller head) learns to predict the human's
controller 18 frames after each frame it sees. Batches are 512 trajectories × 80
frames, carrying the LSTM state between chunks as in a live game.

```
python train_bc.py --run-dir runs/bc_fox --max-hours 10              # one night
python train_bc.py --run-dir runs/bc_fox --resume --max-hours 10     # continue
tensorboard --logdir runs
```

Flags: `--data-dir --run-dir --resume --max-hours --delay --batch-size --unroll --lr
--width --layers --eval-minutes --val-batches --log-every --seed`. Every
`--eval-minutes` it validates and saves `last.pt` (resumable) and, if better, `best.pt`.
TensorBoard: `train/loss`, `train/<head>_loss`, `train/<head>_acc`, `val/loss`,
`val/<head>_acc` for the heads buttons, stick_x, stick_y, cstick.

## Reinforcement learning (`train_rl.py`)

PPO starting from the imitation policy (`runs/bc_fox/bc_fox_18h.pt`), which stays the
"teacher": every frame the loss includes `--kl-coef` × KL(policy ‖ teacher), so the bot
improves where it pays (taking stocks, not dying) while still playing like a person.
Of the 10 games played at once, 6 are self-play against the current policy or an
earlier snapshot (`--latest-prob`, new opponent every `--opponent-minutes`), 2 against
the Ganondorf bot and 2 against the level 9 CPU Fox, all on Final Destination with the
18-frame delay on both Foxes.

```
python train_rl.py --run-dir runs/rl_fox --ganondorf C:\path\to\ganondorf_agent\model\ganondorf_edge_v4.zip --max-hours 10
python train_rl.py --run-dir runs/rl_fox --ganondorf C:\path\to\ganondorf_agent\model\ganondorf_edge_v4.zip --resume --max-hours 10
tensorboard --logdir runs
```

- **Reward** (`smash_rl/reward.py`): ±1 per stock taken / lost, ±0.01 per percent dealt /
  taken, -0.02 per ledge grab, -0.001 per frame offstage after 3 s, +0.0002 per unit
  moved towards the opponent (the small terms block known self-play exploits: planking,
  stalling and running away).
- **Learning** (`smash_rl/ppo.py`): rollouts of 512 frames per game, 2 passes in 4
  minibatches of 128-frame sequences that start from the LSTM state acting had; the value
  network starts from the imitation core and trains alone for the first 30 updates;
  discount 0.997 per frame (half-life ~4 s), GAE λ 0.98, clip 0.2, Adam 3e-5.
- **Acting** replays one CUDA graph per frame for all games (`smash_rl/agent.py`),
  ~0.8 ms instead of ~3.7 ms.
- **Saved:** `last.pt` every 10 minutes and on exit (Ctrl+C is safe), and
  `snapshots/fox_rl_<hours>h.pt` every 20 minutes, which `evaluate.py` and `play.py` take
  like any checkpoint (and which become self-play opponents).
- **Log line** every 10 updates: hours, frames and frames/s (with the share of time in
  the games, acting and learning), KL to the human policy, the share of clipped
  updates, the value network's explained variance and win rates over the last 50 games
  per opponent group. TensorBoard: `train/*` (losses, `teacher_kl`, `entropy`,
  `clip_frac`, `reward_per_frame`), `games/<group>_win_rate`, `games/<group>_stock_diff`,
  `speed/*`.

Flags: `--run-dir --resume --max-hours --init --teacher --n-envs --ganondorf-envs --cpu-envs
--ganondorf --cpu-character --cpu-level --stage --delay --base-port --latest-prob
--opponent-minutes --snapshot-minutes --rollout --chunk --epochs --minibatches --lr
--value-lr --gamma --lam --clip --kl-coef --entropy-coef --value-warmup --kill-reward
--damage-reward --ledge-penalty --approach-reward --stall-penalty --stall-seconds
--log-every --save-minutes --seed`.

## Evaluate and watch (`evaluate.py`)

```
python evaluate.py runs/bc_fox/best.pt --games 20 --n-envs 4 --replays runs/bc_fox/eval
python evaluate.py runs/bc_fox/best.pt --games 2 --n-envs 1 --watch
```

Flags: `--games --n-envs --opponent --cpu-level --ganondorf --versus --stage --delay
--temperature --watch --replays --base-port`. `--versus CHECKPOINT` plays another trained
bot (port 2, same delay), e.g. a reinforcement-learning snapshot against the imitation model. Prints each game and the win rate; `--replays` saves Slippi
replays (open them in the Slippi Launcher, or Dolphin with the same ISO).

**Against a friend's Ganondorf bot** (`ganondorf_agent`, a Stable-Baselines3 PPO model trained
on Final Destination against level 9 CPUs): `--ganondorf` takes its model `.zip` (with its
`meta.json` alongside) and puts it on port 2 instead of the CPU. It decides every 3 frames
with no reaction delay; `smash_rl/ganondorf.py` runs its network with numpy and plays exactly
like the original.

```
python evaluate.py runs/bc_fox/bc_fox_18h.pt --ganondorf C:\path\to\ganondorf_agent\model\ganondorf_edge_v4.zip --games 2 --n-envs 1 --watch
```

## Our Fox in another libmelee bot's match script (`smash_rl/fox_bot.py`)

`FoxBot` drives one port of any libmelee match loop, like the Ganondorf bot does:
`act(gamestate, controller)` every in-game frame before `console.step()`, `reset()`
between games. It reacts 18 frames late, as in training. The shared model is
`models/fox_rl_3h.pt` (after 3 hours of reinforcement learning). It needs this repo,
PyTorch, numpy<2 and libmelee 0.47.3. For the Ganondorf agent's `match.py`, replace
its `friend_agent.py` with:

```python
import sys
sys.path.insert(0, r"C:\path\to\Super_smash_RL")  # this repo
from smash_rl.fox_bot import FoxBot


class FriendAgent(FoxBot):
    def __init__(self, port, opponent_port):
        super().__init__(port=port, opponent_port=opponent_port)  # models/fox_rl_3h.pt
```

and run e.g. `python match.py --dolphin ... --iso ... --p1 ganondorf --p2 friend --watch`.
On Windows, `evaluate.py --ganondorf` (above) plays the same matchup with this repo's
Dolphin fixes.

## Play against the bot (`play.py`)

Plug in an Xbox controller, then:

```
python play.py runs/bc_fox/bc_fox_10h.pt
python play.py runs/bc_fox/bc_fox_10h.pt --stage BATTLEFIELD
```

The bot is port 1 (Fox, 18-frame reaction delay); you are port 2. Pick your character
on the character select screen and press Start; the bot picks the stage. Normal speed
with sound, pausing works, and games repeat until you close the Dolphin window or press
Ctrl+C. Controls (default mapping, kept in `dolphin/play_home`, change them in Dolphin
under Controllers > Port 2 > Configure, `--remap` restores the default): left stick =
control stick, right stick = C-stick, A / B / X / Y as labelled, RB = Z, LT / RT = L / R,
Menu = Start. Flags: `--stage --remap --delay --temperature --port`.

## Results (Phase 1, imitation learning)

Validation loss 0.971 → 0.773 after 10 h, 0.761 after 18 h (plateaued; accuracy: buttons 93%,
stick x 89%, stick y 94%, C-stick 99%). Against the level 9 CPU Fox on Final Destination, 20 games
each, and play style against 22 master-level human Fox players (Fox dittos on FD):

| | 10 h model | 18 h model (`bc_fox_18h.pt`) | Humans |
|---|---|---|---|
| Wins vs CPU 9 | 18 / 20 | **19 / 20** | - |
| Mean stock difference | +1.50 | +2.25 | - |
| Attacks per minute | 47.9 | 50.8 | 46.1 |
| Move variety (entropy, bits) | 3.65 (top: shine 15%) | 3.55 (top: grab 17%) | 3.40 |
| Wavedashes per minute | 4.0 | 3.9 | 4.2 |
| Shines per minute | 7.2 | 7.9 | 8.7 |
| Hits per combo / longest combo | 2.39 / 24 | 2.67 / 33 | 2.76 / 44 |
| L-cancel rate | 91% | 89% | - |
| Damage dealt per minute | 190 | 202 | 148 |

## Files

| File | Purpose |
|---|---|
| `smash_rl/dolphin.py` | One Dolphin through libmelee, with the Windows fixes |
| `smash_rl/state.py` | Raw frame format; `from_peppi` (replays), `from_libmelee` (live) |
| `smash_rl/codec.py` | Controller classes and model inputs |
| `smash_rl/env.py` | Per-frame Gym env with the delay queue; worker process loop |
| `smash_rl/vec_env.py` | Parallel envs in lockstep |
| `smash_rl/replays.py` | Replay filtering and conversion |
| `smash_rl/data.py` | Imitation batches |
| `smash_rl/model.py` | The policy network |
| `smash_rl/agent.py` | Runs a policy live in many games |
| `smash_rl/ganondorf.py` | The Ganondorf bot as a port 2 opponent |
| `smash_rl/fox_bot.py` | Our Fox for other libmelee match scripts |
| `models/` | Shared trained models |
| `poster/` | Course poster (A2, LaTeX): `make_figures.py` (figures and numbers, `--evaluate SNAPSHOT` plays the evaluation games first), `capture_combos.py` (screenshots of the bot's combos from the Dolphin window), `build.ps1` |
| `smash_rl/reward.py` | Phase 2 reward |
| `smash_rl/ppo.py` | PPO with the KL penalty to the imitation policy, value network |
| `train_rl.py` | Reinforcement learning script |
| `prepare_data.py`, `train_bc.py`, `evaluate.py`, `play.py`, `test_env.py` | Scripts above |
| `tests/` | pytest unit tests; `tests/live/` checks that need Dolphin |
| `setup.ps1` / `setup.sh`, `windows_firewall.ps1` | Setup |

## Known issues

- A Dolphin that dies mid-game only ends that game, but one that can't be restarted
  (3 tries) ends a parallel run.
- The CPU can't be made to pick Sheik; a CPU Zelda can still transform mid-game.
- After updating Dolphin, delete `dolphin\instances\` (the patch refuses unknown builds).

## Credits

[libmelee](https://github.com/vladfi1/libmelee) and
[slippi-ai](https://github.com/vladfi1/slippi-ai) by vladfi1 (the imitation-then-RL recipe
and model design followed here), [Project Slippi](https://github.com/project-slippi),
[peppi-py](https://github.com/hohav/peppi-py), the
[ranked replay dataset](https://huggingface.co/datasets/erickfm/melee-ranked-replays), and
the original [DML_RL_SmashBros](https://github.com/KungAlpaca/DML_RL_SmashBros).
Super Smash Bros. Melee is a Nintendo / HAL Laboratory game; this repo contains no game files.
