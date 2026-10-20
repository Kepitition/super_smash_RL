"""Sanity check for the per-frame env: start Dolphin, check inputs arrive, measure speed.

Usage:
    python test_env.py                       # one game window
    python test_env.py --n-envs 10           # 10 games in parallel, as in training
    python test_env.py --stage BATTLEFIELD --cpu-level 9
"""

import argparse
import time

import numpy as np

from smash_rl.codec import ACTION_SIZES, NEUTRAL
from smash_rl.state import PLAYER_DTYPE
from smash_rl.vec_env import make_vec_env


def main():
    p = argparse.ArgumentParser(description="Check the per-frame env and measure its speed")
    p.add_argument("--n-envs", type=int, default=1, help="games in parallel, one worker process and Dolphin each")
    p.add_argument("--frames", type=int, default=3000, help="random-controller frames per game")
    p.add_argument("--cpu-level", type=int, default=3)
    p.add_argument("--opponent", default="FOX", help="CPU character, or RANDOM")
    p.add_argument("--stage", default="FINAL_DESTINATION", help="stage name, e.g. BATTLEFIELD")
    p.add_argument("--delay", type=int, default=18, help="frames between a decision and its input")
    p.add_argument("--port", type=int, default=51441, help="Slippi port of the first game")
    p.add_argument("--headless", action="store_true", help="no window and fast-forward mode (Linux only)")
    args = p.parse_args()

    n = args.n_envs
    print(f"starting {n} dolphin(s) on ports {args.port}-{args.port + n - 1} and navigating menus...")
    t = time.time()
    env = make_vec_env(n_envs=n, base_port=args.port, opponent=args.opponent, cpu_level=args.cpu_level,
                       stages=(args.stage,), delay=args.delay, headless=args.headless)
    try:
        obs, _ = env.reset()
        assert obs["players"].dtype == PLAYER_DTYPE and obs["players"].shape == (n, 2)
        print(f"in game after {time.time() - t:.1f}s | first frame {obs['frame'].tolist()} | stage {obs['stage'].tolist()}")
        # Nobody can move during the countdown (frames < 0); then hold the stick right: every
        # agent must move right once the delay has passed.
        while (obs["frame"] < 0).any():
            obs, _, _, _ = env.step(np.tile(NEUTRAL, (n, 1)))
        x0 = obs["players"]["x"][:, 0].copy()
        for _ in range(args.delay + 60):
            obs, _, _, _ = env.step(np.tile([0, 16, 8, 0], (n, 1)))
        stuck = [i for i in range(n) if obs["players"]["x"][i, 0] - x0[i] < 20]
        print(f"inputs: {n - len(stuck)}/{n} agents moved right" + (f", not moving: games {stuck}" if stuck else ""))
        rng = np.random.default_rng(0)
        start, games = time.time(), 0
        for _ in range(args.frames):
            actions = np.stack([rng.integers(0, size, n) for size in ACTION_SIZES], axis=1)
            obs, _, dones, _ = env.step(actions)
            games += int(dones.sum())
        elapsed = time.time() - start
        print(f"random controllers: {n * args.frames} frames, {games} finished games | "
              f"{n * args.frames / elapsed:.0f} frames/s in total, {args.frames / elapsed:.0f} per game "
              f"(real time is 60)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
