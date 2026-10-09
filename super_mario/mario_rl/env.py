"""Super Mario Bros (NES) environment for PPO training with Stable-Baselines3.

The emulator package (gym-super-mario-bros / nes-py) uses the legacy gym API,
so `MarioEnv` adapts it to Gymnasium and applies the usual Atari-style
preprocessing: frame skipping, grayscale, 84x84 resize and reward scaling.
Frame stacking is done on the vectorized env (VecFrameStack) in `make_vec_env`.
"""

import warnings

warnings.filterwarnings("ignore", module="gym")

import cv2
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

ACTION_SETS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}

FRAME_SIZE = 84
FLAG_BONUS = 50.0  # extra reward (before scaling) for touching the flag pole


class MarioEnv(gym.Env):
    """Gymnasium adapter around the NES Super Mario Bros emulator.

    Observations: uint8 grayscale frames of shape (84, 84, 1).
    Episodes end when Mario reaches the flag or dies (the emulator ends
    single-level episodes on the first death), or are truncated after
    `max_no_progress_steps` agent steps without forward progress.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        world: int = 1,
        stage: int = 1,
        actions: str = "simple",
        frame_skip: int = 4,
        reward_scale: float = 0.1,
        max_no_progress_steps: int = 300,
    ):
        super().__init__()
        self.world, self.stage = world, stage
        self.frame_skip = frame_skip
        self.reward_scale = reward_scale
        self.max_no_progress_steps = max_no_progress_steps

        self._env = JoypadSpace(
            SuperMarioBrosEnv(rom_mode="vanilla", target=(world, stage)),
            ACTION_SETS[actions],
        )
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(FRAME_SIZE, FRAME_SIZE, 1), dtype=np.uint8
        )
        self._raw_frame = None
        self._best_x = 0
        self._no_progress = 0

    @staticmethod
    def _process(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        return small[:, :, None]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(seed)
        frame = self._env.reset()
        # One no-op step to populate info (x_pos, ...).
        frame, _, _, info = self._env.step(0)
        self._raw_frame = frame
        self._best_x = info["x_pos"]
        self._no_progress = 0
        return self._process(frame), info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        prev_frame = self._raw_frame
        for _ in range(self.frame_skip):
            prev_frame = self._raw_frame
            frame, reward, done, info = self._env.step(int(action))
            self._raw_frame = frame
            total_reward += reward
            if info["flag_get"]:
                total_reward += FLAG_BONUS
                terminated = True
                break
            if done:
                terminated = True
                break

        # Max-pool the last two frames to remove NES sprite flicker.
        obs_frame = np.maximum(self._raw_frame, prev_frame) if prev_frame is not None else self._raw_frame

        if info["x_pos"] > self._best_x:
            self._best_x = info["x_pos"]
            self._no_progress = 0
        else:
            self._no_progress += 1
            if not terminated and self._no_progress >= self.max_no_progress_steps:
                truncated = True

        info["max_x_pos"] = self._best_x
        return (
            self._process(obs_frame),
            total_reward * self.reward_scale,
            terminated,
            truncated,
            info,
        )

    def render(self):
        return self._raw_frame

    def close(self):
        self._env.close()


def make_env(rank: int = 0, seed: int = 0, **env_kwargs):
    """Return a thunk that builds one monitored Mario env (for VecEnv workers)."""

    def _init():
        env = MarioEnv(**env_kwargs)
        env = Monitor(env, info_keywords=("flag_get", "max_x_pos"))
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_env(n_envs: int = 8, seed: int = 0, frame_stack: int = 4, subprocess: bool = True, **env_kwargs):
    """Vectorized, frame-stacked Mario envs ready for an SB3 CnnPolicy."""
    thunks = [make_env(i, seed, **env_kwargs) for i in range(n_envs)]
    if subprocess and n_envs > 1:
        venv = SubprocVecEnv(thunks, start_method="spawn")
    else:
        venv = DummyVecEnv(thunks)
    return VecFrameStack(venv, n_stack=frame_stack, channels_order="last")
