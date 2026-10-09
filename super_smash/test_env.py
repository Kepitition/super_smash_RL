"""Sanity check: start Dolphin, validate the env, play with a random agent and measure speed.

Usage:
    python test_env.py                  # one game window, ~1500 random steps
    python test_env.py --steps 3000 --cpu-level 9
    python test_env.py --headless --opponent RANDOM
"""

import argparse
import time

import numpy as np
from stable_baselines3.common.env_checker import check_env

from smash_rl.env import ACTIONS, SmashEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--cpu-level", type=int, default=3)
    p.add_argument("--port", type=int, default=51441)
    p.add_argument("--opponent", default="FOX", help="CPU character, or RANDOM")
    p.add_argument("--headless", action="store_true", help="no window and fast-forward mode (Linux only)")
    args = p.parse_args()

    env = SmashEnv(opponent=args.opponent, cpu_level=args.cpu_level, port=args.port, headless=args.headless)
    try:
        print("starting dolphin and navigating menus...")
        t = time.time()
        obs, info = env.reset(seed=0)
        print(f"in game after {time.time() - t:.1f}s | obs shape {obs.shape} | {len(ACTIONS)} actions")
        check_env(env, warn=True, skip_render_check=True)
        print("check_env passed")

        obs, info = env.reset()
        steps, episodes, ep_return, returns = 0, 0, 0.0, []
        start = time.time()
        while steps < args.steps:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            ep_return += reward
            steps += 1
            if terminated or truncated:
                returns.append(ep_return)
                episodes += 1
                ep_return_done, ep_return = ep_return, 0.0
                won, opponent = info.get("won", 0), info["opponent"]
                t = time.time()
                obs, info = env.reset()
                print(f"game over vs {opponent}: won={won} return={ep_return_done:.2f} | "
                      f"next game started after {time.time() - t:.1f}s, next opponent {info['opponent']}")
        elapsed = time.time() - start
        print(f"random agent: {steps} steps, {episodes} finished games, return so far {ep_return:.2f}, "
              f"damage dealt {info['damage_dealt']:.0f}% taken {info['damage_taken']:.0f}% "
              f"kills {info['kills']} deaths {info['deaths']}")
        print(f"single-env speed: {steps / elapsed:.0f} agent steps/s "
              f"({steps * env.frame_skip / elapsed:.0f} game frames/s, real time is 60)")
        print("obs range:", float(np.min(obs)), float(np.max(obs)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
