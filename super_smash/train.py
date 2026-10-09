"""Train a PPO agent to play Super Smash Bros. Melee against the in-game CPU.

Usage:
    python train.py                                  # Fox vs Fox CPU on Final Destination, curriculum from CPU level 1
    python train.py --timesteps 200000 --run-name quick_test
    python train.py --opponent RANDOM                # a different random CPU character every game
    python train.py --cpu-level 9 --no-curriculum    # always fight a level 9 CPU
    python train.py --gui                            # show a Dolphin window per env (slower)
    python train.py --resume runs/<run>/checkpoints/ppo_smash_update0100_614400steps.zip --cpu-level 4

Monitor with:
    tensorboard --logdir runs
"""

import argparse
import os
import signal
from collections import deque
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from smash_rl.env import CAN_RUN_HEADLESS, make_vec_env


class SmashStatsCallback(BaseCallback):
    """Logs per-game stats to TensorBoard and raises the CPU level when the agent wins often enough."""

    def __init__(self, curriculum: bool, start_level: int, window: int, promote_at: float, max_level: int = 9):
        super().__init__()
        self.curriculum = curriculum
        self.level = start_level
        self.window = window
        self.promote_at = promote_at
        self.max_level = max_level
        self.recent = deque(maxlen=window)
        self.games = 0

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_cpu_level", self.level)
        self.logger.record("smash/cpu_level", self.level)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if info.get("episode") is None:
                continue
            self.games += 1
            won = float(info.get("won", 0.0))
            for key in ("won", "stock_diff", "damage_dealt", "damage_taken", "kills", "deaths"):
                self.logger.record_mean(f"smash/{key}", float(info.get(key, 0.0)))
            self.logger.record("smash/games", self.games)
            # Only games played at the current level count towards promotion.
            if int(info.get("cpu_level", self.level)) == self.level:
                self.recent.append(won)

        if (self.curriculum and self.level < self.max_level and len(self.recent) == self.window
                and np.mean(self.recent) >= self.promote_at):
            self.level += 1
            self.recent.clear()
            self.training_env.env_method("set_cpu_level", self.level)
            print(f"\n>>> win rate reached {self.promote_at:.0%}: CPU level is now {self.level}\n")
        self.logger.record("smash/cpu_level", self.level)
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
            path = os.path.join(self.save_dir, f"ppo_smash_update{update:04d}_{self.model.num_timesteps}steps")
            self.model.save(path)
            self.last_saved_update = update
            print(f"saved checkpoint {path}.zip")


def linear_schedule(initial: float):
    return lambda progress_remaining: progress_remaining * initial


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Super Smash Bros. Melee")
    p.add_argument("--character", default="FOX", help="agent character, e.g. FOX, FALCO, MARTH, CPTFALCON")
    p.add_argument("--opponent", default="FOX", help="CPU character, or RANDOM for a new random character every game")
    p.add_argument("--stage", default="FINAL_DESTINATION",
                   help="FINAL_DESTINATION | BATTLEFIELD | DREAMLAND | YOSHIS_STORY | FOUNTAIN_OF_DREAMS | POKEMON_STADIUM")
    p.add_argument("--cpu-level", type=int, default=1, help="starting CPU level (1-9)")
    p.add_argument("--no-curriculum", action="store_true", help="keep the CPU level fixed")
    p.add_argument("--promote-window", type=int, default=20, help="games used to measure the win rate")
    p.add_argument("--promote-at", type=float, default=0.7, help="win rate needed to raise the CPU level")
    p.add_argument("--frame-skip", type=int, default=3, help="game frames per agent action (60 frames = 1s)")
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=6, help="parallel games (one Dolphin each)")
    p.add_argument("--gui", action="store_true",
                   help="show a Dolphin window per env; by default Linux trains headless, which is much faster")
    p.add_argument("--base-port", type=int, default=51441, help="env i uses port base-port + i")
    p.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda (cpu is fastest for this small MLP)")
    p.add_argument("--save-every", type=int, default=10,
                   help="save weights every N PPO updates (one update = n_envs * n_steps env steps)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="path to a saved .zip model to continue training")
    return p.parse_args()


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    # Background jobs started with `&` ignore SIGINT, so restore it and treat
    # SIGTERM / SIGHUP the same way: all stop training, save the model and close Dolphin.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)
    args = parse_args()
    run_name = args.run_name or f"ppo_{args.character.lower()}_vs_{args.opponent.lower()}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    headless = CAN_RUN_HEADLESS and not args.gui
    print(f"run dir: {run_dir} | device: {args.device} | {args.n_envs} {'headless ' if headless else ''}dolphins "
          f"on ports {args.base_port}-{args.base_port + args.n_envs - 1}")

    env = make_vec_env(
        n_envs=args.n_envs, base_port=args.base_port,
        character=args.character, opponent=args.opponent, stage=args.stage,
        cpu_level=args.cpu_level, frame_skip=args.frame_skip, headless=headless,
    )

    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device, tensorboard_log=run_dir)
            print(f"resumed from {args.resume} at {model.num_timesteps} steps")
        else:
            model = PPO(
                "MlpPolicy",
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
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.ReLU),
                tensorboard_log=run_dir,
                seed=args.seed,
                device=args.device,
                verbose=1,
            )

        callbacks = CallbackList([
            UpdateCheckpointCallback(args.save_every, os.path.join(run_dir, "checkpoints")),
            SmashStatsCallback(not args.no_curriculum, args.cpu_level, args.promote_window, args.promote_at),
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
