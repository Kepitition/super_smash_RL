# DML RL: Super Smash Bros. Melee (and Super Mario Bros.)

Deep reinforcement learning agents for two Nintendo games, trained with PPO
(Stable-Baselines3). The main project is the Melee agent, which beats the
game's hardest CPU and that you can play against yourself.

| Project | Game | What the agent sees | Status |
|---|---|---|---|
| [`super_smash/`](super_smash/README.md) | Super Smash Bros. Melee (GameCube, via Slippi Dolphin + libmelee) | 898 numbers read from game memory | Trained model included: wins 20/20 against the level 9 CPU |
| [`super_mario/`](super_mario/README.md) | Super Mario Bros. World 1-1 (NES, via gym-super-mario-bros) | 4 stacked 84x84 grayscale frames | Training code only; an early 1.6M-step run got about halfway through the level, without clearing it |

Each project is self-contained: its own `setup.sh`, `requirements.txt`, Python
environment (`.venv`) and README with full instructions.

## Quick start: play against the Melee agent (Linux)

```bash
cd super_smash
./setup.sh                                          # Python env + emulator downloads
cp /path/to/your.iso "Super Smash Bros Melee v1.02.iso"   # your own NTSC v1.02 ISO
source .venv/bin/activate
python play.py models/fox_vs_fox_cpu9.zip           # Xbox controller; --input keyboard otherwise
```

Watch it beat the CPU instead:

```bash
python evaluate.py models/fox_vs_fox_cpu9.zip --cpu-level 9 --games 3
```

Train your own (headless; the shipped model reached CPU level 9 in 38 minutes with 12 games on 8 cores):

```bash
python train.py --n-envs 8 --timesteps 14000000 --save-every 50 --run-name my_run
```

See [`super_smash/README.md`](super_smash/README.md) for results, controls, training
options and how the agent works.

## Repository layout

```
.
├── README.md            this file
├── AGENTS.md            orientation guide for AI coding assistants (CLAUDE.md points to it)
├── super_smash/         Melee agent
│   ├── smash_rl/        env.py (Gymnasium env around Dolphin), xbox_pad.py (controller input for play.py)
│   ├── models/          trained model(s) shipped with the repo
│   ├── train.py  evaluate.py  play.py  test_env.py  setup.sh  requirements.txt
│   └── (ignored) .venv/  dolphin/  runs/  *.iso
└── super_mario/         Mario agent
    ├── mario_rl/        env.py (Gymnasium adapter + preprocessing)
    ├── train.py  evaluate.py  test_env.py  setup.sh  requirements.txt
    └── (ignored) .venv/  runs/
```

## Not in the repo

- **Game files.** Melee is not included and must never be committed. Each person
  supplies their own legally dumped ISO (`*.iso` and `*.zip` are git-ignored). Super
  Mario Bros. ships inside the `gym-super-mario-bros` package.
- **Emulators.** `super_smash/setup.sh` downloads Slippi Dolphin and the headless
  ExiAI Dolphin build into `super_smash/dolphin/`.
- **Python environments and training runs** (`.venv/`, `runs/`, logs).

## Platforms

The projects were started on macOS (Apple M4) and the Melee project was then moved to
Linux (Ubuntu 22.04), where it gained headless training and `play.py`. Melee
training with Dolphin windows still works on macOS. The Mario project has only been
run on macOS; its `setup.sh` needs [`uv`](https://docs.astral.sh/uv/).
