# Guide for AI coding assistants

A Fox bot for Super Smash Bros. Melee: imitation learning on human replays (Phase 1),
then PPO self-play with a KL leash to the imitation policy (Phase 2), then evaluation
and a play script (Phase 3). Slippi Dolphin + libmelee, developed on Windows 11.
Human docs: `README.md`. Designs: `docs/superpowers/specs/`, plans: `docs/superpowers/plans/`.

## Ground rules

- `.venv` (Python 3.10-3.12; numpy<2 has no wheels for 3.13+). Run from the repo root:
  `.venv\Scripts\python.exe ...` on Windows, `.venv/bin/python ...` elsewhere.
- No git commits unless the user asks (they share the repo themselves).
- Never commit game files (`*.iso`, `*.zip`), `dolphin/`, `.venv/`, `runs/` or data
  (`D:\smash_data`). They are git-ignored or outside the repo.
- Code, comments, CLI help and printed messages are in English. Comments are sparse
  and explain why. Every option is an argparse flag with help; README lists the flags.
- Tests: `python -m pytest` (unit, no Dolphin). Live checks in `tests/live/` and
  `test_env.py` open Dolphin windows; afterwards check no `Slippi Dolphin.exe` is left.

## Map

| Path | Contents |
|---|---|
| `smash_rl/state.py` | `PLAYER_DTYPE` raw frame; `from_peppi` (replays), `from_libmelee` (live); stage indices; `BUTTON_BITS` |
| `smash_rl/codec.py` | Controller classes (`ACTION_SIZES = (48, 17, 17, 5)`, `NEUTRAL`, `controller_classes`, `apply_controller`, `STICK_SEND`); `encode_state` model inputs |
| `smash_rl/dolphin.py` | `Dolphin` (start/stop/next_state), per-port exe copies and patches, buffered `_Controller`, robust `_Console`, `kill_children_on_exit` |
| `smash_rl/env.py` | `SmashEnv` (one frame per step; port 2 CPU / `RANDOM` / `HUMAN` / `POLICY` self-play with (2, 4) actions / `opponent_bot`; `reward` kwargs), `DelayQueue`, `step_and_autoreset`, `env_worker` (no SB3/torch imports) |
| `smash_rl/vec_env.py` | `make_vec_env` → `SubprocVecEnv` / `LocalVecEnv`, `stack_obs`, `_lean_spawn` |
| `smash_rl/replays.py` | Replay filters (`check_game`), `convert_safely`, `read_index`, `game_split` |
| `smash_rl/data.py` | `load_index`, `trajectory_list`, `UnrollBatcher` (stateful unrolls), `Prefetcher`, `to_torch` |
| `smash_rl/model.py` | `Policy` (LSTM + FFW blocks, autoregressive heads), `load_policy` |
| `smash_rl/agent.py` | `Agent`: live batched acting with per-game LSTM state, one CUDA graph replay per frame, `logp` of each decision |
| `smash_rl/reward.py` | `Reward`: kills/damage (zero-sum) + ledge-grab, offstage-stall and approach shaping, from raw frames |
| `smash_rl/ppo.py` | `ValueNet` (imitation core + head), `gae`, `Rollout` (chunk-start LSTM states), `Learner` (PPO + KL(policy ‖ teacher)) |
| `train_rl.py` | RL script: self-play / Ganondorf / CPU games, opponent pool of snapshots, `last.pt`, `snapshots/` |
| `smash_rl/ganondorf.py` | `GanondorfBot`: a friend's SB3 Ganondorf (847→256→256→31 MLP) in numpy, for `SmashEnv(opponent_bot=...)` on port 2 (human-type slot, acts every frame); matches its `PPO.predict` exactly |
| `smash_rl/fox_bot.py` | `FoxBot`: our policy behind the Ganondorf bot's interface (`act(gs, controller)`, `reset()`), own 18-frame queue, for other people's libmelee loops |
| `models/` | Shared models (in git, ~40 MB each, keep it to a few): `fox_rl_3h.pt` |
| `poster/` | DML course poster on the course template (don't change the theme `.sty` files): `poster.tex` takes every number from `fig/results.tex`, written with the figures by `make_figures.py` from `runs/` logs and `runs/poster/` evaluations; screenshots come from `capture_combos.py` (frame-stepped Dolphin + `PrintWindow`; raw frames in git-ignored `poster/frames/`, crops in `make_figures.HERO`/`STRIP`); `build.ps1` (MiKTeX) |
| `prepare_data.py`, `train_bc.py`, `evaluate.py` (`--ganondorf`, `--versus`), `test_env.py` | Scripts |
| `play.py` | Human (Xbox pad, port 2) vs a bot; mapping in `dolphin/play_home/Config/GCPadNew.ini` |
| `dolphin/` | Not in git: `Slippi Dolphin\` (portable 3.6.4) and `instances\slip<port>\` |

## Conventions that must not drift

- Delay `d`: the decision made after observing frame t is the controller of frame t+d.
  Imitation labels are the controller at t+d; the previous-action input is the label
  for t+d-1 (neutral at a trajectory's first frame). Env queue length is d-1.
- Players are always ordered `[agent, opponent]` (slot 1 trajectories are swapped).
- Raw frames follow libmelee's conventions (truncated hitlag/hitstun, libmelee's +1 on
  zero-indexed action frames). Any change to `state.py` must keep
  `tests/live/check_replay_vs_live.py` passing.
- Changing `encode_state`, `ACTION_SIZES` or the delay invalidates trained models.
- Self-play: the port 2 policy sees `players[:, ::-1]` and has its own `delay` queue. RL
  rollouts store each game's first-frame flags; `Policy.forward` takes a (B, T) reset
  mask and restarts the LSTMs there. The first PPO minibatch must reproduce acting
  (ratio 1, `tests/test_ppo.py`); keep acting and training numerically identical (fp32).
- `Agent`'s CUDA graph reads parameters by address: `load_state_dict` and optimizer
  steps keep addresses, but `model.to()` re-packs the cuDNN LSTM weights even on the
  same device (the agent then re-records its graph). Don't `.to()` a model mid-training.

## Windows specifics (learned the hard way)

- **Pipe names are global**: per-port exe copies rename `\\.\pipe\slippibot` to
  `\\.\pipe\slip<port>` (same length) and `_Console.get_dolphin_pipes_path` points
  libmelee at them.
- **Windows Dolphin 3.6.4 does not block on its input pipe**: `UpdateInput` returns when
  a read finds no data, so frames ran on stale input and lag drifted (0-90 frames with
  4+ games). `patch_dolphin_exe` NOPs that `je` (byte pattern `EARLY_RETURN`), giving
  mainline's wait-for-FLUSH loop. Verified: lag 0. Bump `PATCH_VERSION` when patches change.
- Because Dolphin now waits for input, anything that stops stepping mid-game freezes
  it (e.g. the end-of-game screen); `reset()` steps through it.
- It also asked for input on some menu/loading frames that send no event, deadlocking with
  libmelee: the console runs in polling mode and re-sends the input after `STALL_SECONDS`
  (1 s) without a frame. `next_state` reports a closed window at once and never times
  out with a person on port 2 (pauses).
- **Inputs must be written in one go** per frame (`_Controller` buffers until flush).
- `Console.stop()` could hang (kill Dolphin first); a failed `connect()` couldn't be
  retried (new `SlippstreamClient`); many Dolphins starting together take ~30 s.
- Each process joins a kill-on-close job, so workers and Dolphins die with the main process.
- Workers stay PyTorch-free; `_lean_spawn` hides `__main__` while spawning.
- Instant Match restarts never finalize a Slippi replay, so `save_replays` disables them.
- The libmelee game-start flush makes the very first frame of a game use neutral input.
- Only OGL, Vulkan and Software renderers exist in this build; Vulkan crashed with 20 games.
- Speed: ~1200 frames/s in total from 10 games up; 10 games is the sweet spot.

## Data facts

- The ranked dumps have no game-end record (`game.end is None`); "finished" means a
  player at 0 stocks on the last frame or the 8-minute timer reached.
- Z reads back as a light trigger press in the processed trigger, so `controller_classes`
  ignores the analog trigger when Z is pressed; classes "Z + light" never occur in data.

## Known open issues

- A Dolphin that dies or freezes mid-`step()` ends only that game (truncated, `info["crashed"]`);
  a reset that fails 3 times still ends a vectorized run.
- RL speed: ~400 frames/s with 10 games while the machine was busy (random inputs got 520
  then, 1200 on an idle machine); ~25% of the time is acting (GPU shared with the Dolphins).
- After updating Dolphin, delete `dolphin\instances\`.
