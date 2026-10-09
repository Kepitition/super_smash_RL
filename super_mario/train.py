"""Train a PPO agent to play Super Mario Bros with Stable-Baselines3.

Usage:
    python train.py                              # World 1-1, 8 parallel envs, 10M steps
    python train.py --timesteps 200000 --run-name quick_test
    python train.py --save-every 5                # save weights every 5 PPO updates
    python train.py --resume runs/<run>/checkpoints/ppo_mario_update0100_409600steps.zip

Monitor with:
    tensorboard --logdir runs
"""

import argparse
import os
import signal
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from mario_rl.env import make_vec_env


class MarioStatsCallback(BaseCallback):
    """Logs Mario-specific episode stats (distance reached, level clears) to TensorBoard."""

    def __init__(self):
        super().__init__()
        self.total_clears = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is None:
                continue
            cleared = bool(info.get("flag_get", False))
            self.total_clears += cleared
            self.logger.record_mean("mario/max_x_pos", info.get("max_x_pos", 0))
            self.logger.record_mean("mario/clear_rate", float(cleared))
            self.logger.record("mario/total_clears", self.total_clears)
        return True


class UpdateCheckpointCallback(BaseCallback):
    """Saves the model every `save_every` PPO updates (rollout + optimization epochs)."""

    def __init__(self, save_every: int, save_dir: str):
        super().__init__()
        self.save_every = save_every
        self.save_dir = save_dir
        self.last_saved_update = None

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # Called right after the previous update finished training.
        steps_per_update = self.model.n_steps * self.model.n_envs
        update = self.model.num_timesteps // steps_per_update
        if update > 0 and update % self.save_every == 0 and update != self.last_saved_update:
            path = os.path.join(self.save_dir, f"ppo_mario_update{update:04d}_{self.model.num_timesteps}steps")
            self.model.save(path)
            self.last_saved_update = update
            print(f"saved checkpoint {path}.zip")


def linear_schedule(initial: float):
    return lambda progress_remaining: progress_remaining * initial


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Super Mario Bros")
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--actions", choices=["right_only", "simple", "complex"], default="simple")
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--save-every", type=int, default=10,
                   help="save weights every N PPO updates (one update = n_envs * n_steps env steps)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="path to a saved .zip model to continue training")
    return p.parse_args()


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    # Background jobs started with `&` ignore SIGINT, so restore it and treat
    # SIGTERM the same way: both stop training and save the model.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    args = parse_args()
    run_name = args.run_name or f"ppo_w{args.world}s{args.stage}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    device = pick_device(args.device)
    print(f"run dir: {run_dir} | device: {device}")

    env = make_vec_env(
        n_envs=args.n_envs, seed=args.seed,
        world=args.world, stage=args.stage, actions=args.actions,
    )

    if args.resume:
        model = PPO.load(args.resume, env=env, device=device, tensorboard_log=run_dir)
        print(f"resumed from {args.resume} at {model.num_timesteps} steps")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=linear_schedule(args.lr),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=run_dir,
            seed=args.seed,
            device=device,
            verbose=1,
        )

    callbacks = CallbackList([
        UpdateCheckpointCallback(args.save_every, os.path.join(run_dir, "checkpoints")),
        MarioStatsCallback(),
    ])

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="PPO",
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nstopped - saving current model")
    finally:
        final_path = os.path.join(run_dir, "final_model")
        model.save(final_path)
        print(f"saved {final_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
