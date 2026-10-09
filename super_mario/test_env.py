"""Sanity check: validate the env, run a random agent, and measure emulator speed.

Usage:
    python test_env.py
"""

import time

import cv2
import numpy as np
from stable_baselines3.common.env_checker import check_env

from mario_rl.env import MarioEnv


def main():
    env = MarioEnv(world=1, stage=1)
    check_env(env, warn=True)
    print("check_env passed")
    print("action space:", env.action_space, "| observation space:", env.observation_space)

    obs, info = env.reset(seed=0)
    steps, episodes, ep_return, returns, best_x = 0, 0, 0.0, [], 0
    start = time.time()
    while steps < 2000:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ep_return += reward
        best_x = max(best_x, info["x_pos"])
        steps += 1
        if terminated or truncated:
            returns.append(ep_return)
            episodes += 1
            ep_return = 0.0
            obs, info = env.reset()
    elapsed = time.time() - start

    print(f"random agent: {steps} steps, {episodes} episodes, "
          f"mean return {np.mean(returns) if returns else ep_return:.2f}, furthest x_pos {best_x}")
    print(f"single-env speed: {steps / elapsed:.0f} agent steps/s "
          f"({steps * env.frame_skip / elapsed:.0f} emulator frames/s)")

    cv2.imwrite("sample_raw_frame.png", cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
    cv2.imwrite("sample_processed_frame.png", cv2.resize(obs[:, :, 0], (252, 252), interpolation=cv2.INTER_NEAREST))
    print("saved sample_raw_frame.png and sample_processed_frame.png")
    env.close()


if __name__ == "__main__":
    main()
