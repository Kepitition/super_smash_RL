# Guide for AI coding assistants

Two independent PPO (Stable-Baselines3) projects. `super_smash/` (Super Smash Bros.
Melee through Slippi Dolphin + libmelee) is the main one and gets most changes;
`super_mario/` (NES Super Mario Bros. World 1-1) is a separate, image-based agent.
Human docs: `README.md`, `super_smash/README.md`, `super_mario/README.md`.

## Ground rules

- Each project has its own `.venv` (Python 3.10). Run commands from inside the project
  folder with that venv: `cd super_smash && .venv/bin/python ...`.
- Never commit game files (`*.iso`, `*.zip` except `super_smash/models/*.zip`),
  emulators (`super_smash/dolphin/`), `.venv/` or `runs/`. They are git-ignored; keep it so.
- Code, comments, CLI help and printed messages are in English. Comments are sparse
  and explain why, not what. Every option is an argparse flag with a help string, and
  the README tables list the flags; keep them in sync when you change behaviour.
- There is no test suite. Verify changes by running the scripts (below), headless
  where possible, and check that no Dolphin is left running afterwards.

## super_smash map

| Path | Contents |
|---|---|
| `smash_rl/env.py` | `SmashEnv` (Gymnasium env), `ACTIONS`, observation layout, `make_vec_env`, `NonDaemonSubprocVecEnv`, `SmashMonitor` |
| `smash_rl/xbox_pad.py` | `XboxPad`: reads an Xbox pad from evdev in a thread and drives port 2 through a libmelee pipe; `find_pad()` |
| `train.py` | PPO setup, `SmashStatsCallback` (TensorBoard `smash/*` stats + curriculum), `UpdateCheckpointCallback` |
| `evaluate.py` | Plays N games per model (or per `.zip` in a folder, writing `eval_results.csv`) |
| `play.py` | Human vs model. `HumanEnv` = `SmashEnv` + `XboxPad`; keyboard mapping written to Dolphin's `GCPadNew.ini` |
| `test_env.py` | One env, `check_env`, random actions, speed |
| `setup.sh` | venv (uv or python3.10) + downloads Slippi Dolphin 3.6.4 and, on Linux, the ExiAI headless build |
| `models/fox_vs_fox_cpu9.zip` | Shipped model (898-dim observation), see README results |
| `Super Smash Bros Melee v1.02.iso` | Not in git; each user provides it (`DEFAULT_ISO`) |
| `dolphin/` | Not in git: `Slippi_Online-x86_64.AppImage` (windowed), `Slippi_Online-x86_64-ExiAI.AppImage` (headless), `Slippi Dolphin.app` (macOS), `play_home/` (persistent Dolphin user folder for `play.py`) |

### How SmashEnv works

- One Dolphin process per env, controlled by libmelee. Port 1 = agent, port 2 =
  opponent. Each env uses its own Slippi port (`train.py`: 51441 + i, `evaluate.py`:
  51500, `play.py`: 51520); use other ports when testing in parallel.
- An episode is a whole game (4 stocks). `reset()` starts or restarts Dolphin as needed
  and walks the menus with `melee.MenuHelper` (`_go_to_game`); `step()` holds one of the
  31 `ACTIONS` for `frame_skip` (3) frames and sums the per-frame reward.
- Opponent modes (`opponent=`):
  - a character name: the CPU plays it at `cpu_level`;
  - `"RANDOM"`: a new CPU character from `RANDOM_OPPONENTS` (25, no Sheik/Nana) every game;
  - `"HUMAN"`: port 2 is a person; no CPU pad, no menu timeout, non-blocking input, sound
    on, bigger window, `SIDevice1` set to a standard controller.
- `headless=True` (Linux): ExiAI AppImage, `gfx_backend="Null"`, EXI inputs, fast-forward.
- `instant_restart` (libmelee's Instant Match gecko code, patched to keep the stage) is
  only used for a fixed CPU opponent with windows. Otherwise games go back through the menus.
- `dolphin_home=None` uses a fresh temporary Dolphin user folder per start;
  `play.py` passes `dolphin/play_home` so controller mappings persist.
- Curriculum: `SmashStatsCallback` calls `env_method("set_cpu_level")` after 20 games at
  >= 70% wins. With instant restarts a level change restarts Dolphin on the next reset.

### Observation (898 float32, clipped to [-10, 10])

Per player block of 447 (`PLAYER_SIZE`): agent at `[0:447]`, opponent at `[447:894]`, then
4 globals at `[894:898]` (dx/100, dy/100, distance/100, frame/28800).

| Offset in block | Size | Content |
|---|---|---|
| 0 | 20 | x/100, y/100, percent/100, stock/4, facing (+1 right / -1 left), on_ground, off_stage, invulnerable, min(hitlag,30)/10, min(hitstun,120)/30, jumps_left/2, shield/60, speed_air_x/3, speed_ground_x/3, speed_y/3, clip(knockback_x/5, ±5), clip(knockback_y/5, ±5), min(action_frame,120)/60, (edge - abs(x))/100, iasa |
| 20 | 27 | character one-hot, Melee internal id 0..26 (`NUM_CHARACTERS`) |
| 47 | 400 | action-state one-hot (`NUM_ANIMATIONS`; ids >= 400 map to 399) |

Changing anything here, `ACTIONS` or `frame_skip` makes saved models unusable
(`PPO.load` with an env fails; `play.py` checks the shape). Retrain and replace the
model in `models/` and the README results if you do. The 844-dim layout without the
character one-hot is obsolete.

Reward per frame: `damage_scale (0.01) x (damage dealt - damage taken) + kills - deaths`.
Info keys: `damage_dealt`, `damage_taken`, `kills`, `deaths`, `cpu_level`, `opponent`,
and at game end `won`, `stock_diff` (logged via `SmashMonitor` / `INFO_KEYWORDS`).

### Commands to verify changes

```bash
cd super_smash
.venv/bin/python test_env.py --headless --steps 3000                  # env end to end, ~20 s
.venv/bin/python test_env.py --headless --opponent RANDOM --cpu-level 9
.venv/bin/python evaluate.py models/fox_vs_fox_cpu9.zip --headless --games 5 --port 51530
.venv/bin/python train.py --n-envs 2 --timesteps 20000 --run-name smoke --base-port 51560
pgrep -af "^/tmp/.mount_Slippi" || echo "no Dolphin left"              # leftovers
```

Windowed runs (`test_env.py` without `--headless`, `evaluate.py`, `play.py`) open
Dolphin windows on the user's screen. Kill leftover Dolphins by PID from the
`pgrep` above; avoid `pkill -f <pattern>`, which also matches your own shell command.

### Things learned the hard way

- **ExiAI + Instant Match crashes** Melee when a game ends ("Invalid read" spam, then
  the ENet connection drops). Headless therefore always goes through the menus.
- **Blocking input deadlocks on pause**: Dolphin waits for the bot every frame but no
  game states arrive while paused. `HUMAN` mode uses non-blocking input.
- **libmelee writes ini files with lower-cased keys** (ConfigParser default). Edit
  Dolphin ini files the same way (no `optionxform = str`), or a persistent Dolphin
  folder ends up with duplicate keys and the next run raises `DuplicateOptionError`.
- `MenuHelper` remembers `stage_selected` between visits; `_go_to_game` makes new
  helpers on every call.
- The CPU can't be forced to pick Sheik (libmelee raises); a CPU Zelda can still
  transform mid-game, which the character one-hot shows as Sheik (id 7).
- Stick values: libmelee's `Controller` (`fix_analog_inputs=True`) maps
  `tilt_analog_unit` +-1 to Melee's internal +-1 (raw +-80). Melee thresholds: deadzone
  0.2875, tap jump 0.6625; `xbox_pad.py` caps up at 0.6 to disable tap jump.
- Melee action ids >= 0x155 are character specific (mostly special moves).
- `iasa` is always 0: libmelee never sets it from game data.

### Known open issues

- Memory: about 600 MB per env (Dolphin ~410 MB + worker ~190 MB). A 12-env run on a
  14 GB laptop hit the OOM killer after 2 h. Recommend `--n-envs 8` or fewer.
- A Dolphin dying mid-`step()` raises in the worker (`EnetDisconnected`) and ends the
  whole training run; `final_model.zip` is still saved, but `env.close()` then raises
  `EOFError` and the process hangs on the remaining non-daemon workers.
- On a CPU level change in menu mode (headless or RANDOM), each env gets stuck once on
  the character select screen for `max_menu_frames` (6000) and recovers by restarting
  Dolphin ("restarting dolphin after: stuck in menus").
- The agent only knows Final Destination and Fox (it plays other CPU characters fine).

## super_mario map

Image-based PPO on World 1-1 with `gym-super-mario-bros` / `nes-py` (legacy gym 0.26 API,
pinned with `numpy<2`). `mario_rl/env.py` adapts it to Gymnasium (frame skip 4,
max-pool, grayscale 84x84, frame stack 4 via `VecFrameStack`, reward scaled by 0.1 with
a flag bonus); `train.py` uses `CnnPolicy` and picks MPS/CUDA/CPU. It has only been run
on macOS (Apple M4); `setup.sh` requires `uv`. See `super_mario/README.md`.
