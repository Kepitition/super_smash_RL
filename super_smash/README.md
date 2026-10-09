# Super Smash Bros. Melee PPO Agent

Trains a PPO agent (Stable-Baselines3) to play Super Smash Bros. Melee (NTSC v1.02)
as Fox against the game's own CPU, through the Slippi Dolphin emulator and
[libmelee](https://github.com/vladfi1/libmelee). The agent reads the game state
straight from memory (positions, damage, animations...), not pixels.

The repo ships a trained model, `models/fox_vs_fox_cpu9.zip`, that beats the
level 9 CPU, and `play.py` lets you play against it with an Xbox controller or
the keyboard.

## Results

`models/fox_vs_fox_cpu9.zip` was trained on Linux (Ryzen 7 7840HS, 8 cores; the GPU
is not needed) with 12 headless games in parallel. The curriculum reached CPU
level 9 after 860k steps (38 minutes); training stopped at 4.6M steps (1 h 50 min).

| Evaluated against (Final Destination) | Wins | Mean stock difference |
|---|---|---|
| CPU level 9 Fox | 20 / 20 | +3.0 |
| CPU level 9 Fox, sampled actions (`--stochastic`) | 20 / 20 | +2.6 |
| CPU level 9, a random character every game (never seen in training) | 18 / 20 | +2.1 |
| CPU level 5 Fox | 10 / 10 | +3.5 |
| CPU level 1 Fox | 10 / 10 | +3.9 |

The Melee CPU is predictable, so expect the agent to be much easier to beat for a person.

## Requirements

- **Linux x86-64** for headless training and `play.py`. macOS (where the project
  started) can still train, evaluate and watch with Dolphin windows.
- Python 3.10.
- **Your own Melee NTSC v1.02 ISO**, not included in the repo. Put it in this folder as
  `Super Smash Bros Melee v1.02.iso` (game ID `GALE01`, revision 2).
- About 600 MB of RAM per parallel game (one Dolphin plus one Python worker each).

## Setup

```bash
./setup.sh                    # .venv with Python 3.10 + dependencies, downloads Dolphin
cp /path/to/your.iso "Super Smash Bros Melee v1.02.iso"
source .venv/bin/activate
python test_env.py --headless # starts one game and plays random actions
```

`setup.sh` uses `uv` when it is installed and `python3.10 -m venv` otherwise. It
downloads Slippi Dolphin 3.6.4 (the normal emulator, with a window) and, on Linux,
the headless "ExiAI" Dolphin build from libmelee's author, used for training. On
Ubuntu, AppImages need `libfuse2`. PyTorch's CUDA build is not needed; to skip its
~3 GB download, run `pip install torch --index-url https://download.pytorch.org/whl/cpu`
before `pip install -r requirements.txt`.

## Play against the agent

```bash
python play.py models/fox_vs_fox_cpu9.zip                    # Xbox controller plugged in by USB
python play.py models/fox_vs_fox_cpu9.zip --input keyboard
```

The agent is port 1 and plays Fox; you are port 2. Pick your character on the
character select screen and press Start; the agent picks Final Destination, the only
stage it knows. Games run at normal speed with sound and can be paused. Close the
Dolphin window or press Ctrl+C in the terminal to stop.

| Xbox controller | Does |
|---|---|
| Left stick | Move. Pushing up never jumps |
| A | Tap: normal attack (jab, or a tilt with a direction). Hold ~150 ms with a direction: smash attack |
| B | Special (up + B: up special) |
| X / Y | Jump (the only way to jump) |
| Right stick | Smash attacks on the ground, aerials in the air (Melee's C-stick) |
| RB | Z (grab) |
| LT / RT | L / R (shield, air dodge) |
| Menu | Start |

The Xbox controller is read in Python (`smash_rl/xbox_pad.py`) and replayed to the
game, which is what allows the two non-Melee controls above. `--smash-hold-ms` changes
the hold time. Held-A smashes can't be charged; Melee's own smash input (flick the
stick and press A) still can.

| Keyboard | Does |
|---|---|
| W A S D | Control stick (W jumps, like in Melee) |
| Hold Left Shift | Half tilt (walk, tilt attacks) |
| Arrow keys | C-stick |
| J / K | A / B |
| L / I | X / Y (jump) |
| O | Z (grab) |
| Space | R (shield) |
| Enter | Start |

The keyboard mapping lives in Dolphin (`dolphin/play_home`). Change it in the Dolphin
window with Controllers > Port 2 > Configure; `--remap` restores the default.

## Watch or evaluate a model

```bash
python evaluate.py models/fox_vs_fox_cpu9.zip --cpu-level 9 --games 3       # watch, normal speed
python evaluate.py models/fox_vs_fox_cpu9.zip --games 20 --headless         # just the numbers, fast
python evaluate.py models/fox_vs_fox_cpu9.zip --opponent RANDOM --headless  # other characters
python evaluate.py runs/<run>/checkpoints --games 5 --headless --cpu-level 5  # score every checkpoint
```

Passing a folder evaluates every `.zip` in it and writes `eval_results.csv` there.
Actions are deterministic (most likely action) unless `--stochastic` is given.
`--cpu-level` defaults to 9, so set it to the level a model actually reached.

## Train

```bash
python train.py --n-envs 8 --timesteps 14000000 --save-every 50 --run-name my_run
python train.py --n-envs 8 --opponent RANDOM          # a random CPU character every game
python train.py --cpu-level 9 --no-curriculum         # always fight a level 9 CPU
python train.py --resume runs/<run>/checkpoints/<file>.zip --cpu-level 7
python train.py --gui                                 # show a Dolphin window per game (much slower)
```

- On Linux training is headless by default: about 600 steps/s with 6 games and
  740 with 12 on 8 cores, versus about 290 with windows. The bottleneck is the
  emulator, which runs on the CPU; the network takes under 1% of the time, so a GPU doesn't help.
- **Use at most one game per physical core and watch the memory.** A 12-game run on a
  14 GB laptop ran out of memory after two hours and stopped (see Known issues).
- The learning rate decays linearly to 0 over `--timesteps`, so pick the length up
  front rather than stopping early.
- Checkpoints go to `runs/<run>/checkpoints/` every `--save-every` PPO updates (one
  update is `n_envs x 1024` steps), and `runs/<run>/final_model.zip` is written at the
  end or when you press Ctrl+C (once).
- `--resume` does not remember the curriculum level; pass `--cpu-level` again.
- To keep a laptop from sleeping when the lid closes, start training with
  `systemd-inhibit --what=sleep:handle-lid-switch python train.py ...`. Closing the
  terminal stops training (and saves the model).

## Monitor

```bash
tensorboard --logdir runs        # open http://localhost:6006
```

| Tag | Meaning |
|---|---|
| `smash/won` | Fraction of finished games the agent won |
| `smash/cpu_level` | Current curriculum level |
| `smash/stock_diff` | Agent stocks minus CPU stocks at the end of a game |
| `smash/kills`, `smash/deaths`, `smash/damage_dealt`, `smash/damage_taken` | Per game |
| `rollout/ep_rew_mean` | Average reward per game |
| `train/entropy_loss` | Minus the policy entropy; starts at -3.43 (31 equally likely actions) |

## How it works

- **Environment** (`smash_rl/env.py`): each env runs its own Dolphin and talks to it
  through libmelee. The agent is port 1, the CPU port 2, on Final Destination. One
  episode is one full game (4 stocks); the env goes through the menus by itself.
- **Observation:** 898 floats read from game memory, `[agent 447 | opponent 447 | 4 global]`.
  Per player: 20 continuous features (position, damage, stocks, facing, on ground,
  off stage, invulnerable, hitlag and hitstun left, jumps left, shield, self and
  knockback speeds, animation frame, distance to the edge, IASA), a 27-way character
  one-hot and a 400-way action-state one-hot. Global: dx, dy, distance and game time.
- **Actions:** 31 discrete stick + button combinations, each held for 3 frames
  (20 decisions per second): stick only in 9 directions, A + 9 directions, B + 5,
  X (jump) + 3, Z (grab), L (shield, roll left/right, spot dodge).
- **Reward:** per frame, `0.01 x (damage dealt - damage taken) + (stocks taken - stocks lost)`.
- **PPO:** `MlpPolicy` with separate 256-256 ReLU actor and critic networks.
  1024-step rollouts per env, batch 512, 4 epochs, learning rate 3e-4 decaying to 0,
  gamma 0.995, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01. All are flags
  of `train.py`.
- **Curriculum:** starts against CPU level 1 and moves up one level whenever the agent
  wins at least 70% of the last 20 games at the current level, up to 9.
- **Headless mode:** the ExiAI Dolphin build runs with no window and Melee's
  fast-forward mode. It crashes on the "instant match restart" code used with windows,
  so headless games go back through the menus between games (about 0.6 s).

## Files

| File | Purpose |
|---|---|
| `smash_rl/env.py` | Gymnasium env: Dolphin lifecycle, menus, observation, actions, reward; vectorized envs |
| `smash_rl/xbox_pad.py` | Reads an Xbox controller and replays it to port 2 with the custom controls |
| `train.py` | PPO training with curriculum, checkpoints and TensorBoard stats |
| `evaluate.py` | Watch a model or score a folder of checkpoints |
| `play.py` | Play against a model |
| `test_env.py` | Environment sanity check and speed test |
| `setup.sh` | Python environment and emulator downloads |
| `models/` | Trained models shipped with the repo |

## Known issues

- **Running out of memory stops training.** When a Dolphin dies mid-game the error
  ends the whole run (the model is still saved), and the process can then hang
  instead of exiting; stop it with Ctrl+C or `kill`. Train with fewer games and close
  heavy apps.
- **Changing the CPU level gets each headless game stuck in the menus once.** The env
  notices after about 6,000 frames, restarts that Dolphin and goes on, so it costs a
  few seconds per level change.
- The agent has only seen Final Destination and plays only Fox.
- The `iasa` feature is always 0 (libmelee never fills it in).
- Models trained before the character one-hot was added (844 inputs instead of 898)
  can't be loaded by the current env.

## Credits

[libmelee](https://github.com/vladfi1/libmelee) and its headless
[ExiAI Dolphin build](https://github.com/vladfi1/slippi-Ishiiruka) by vladfi1, and
[Project Slippi](https://github.com/project-slippi/Ishiiruka). Super Smash Bros. Melee
is a Nintendo / HAL Laboratory game; this repo contains no game files.
