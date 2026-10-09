"""Watch a trained PPO agent play and report how often it clears the level.

Usage:
    python evaluate.py runs/<run>/final_model.zip
    python evaluate.py runs/<run>/checkpoints/ppo_mario_update0500_2048000steps.zip --episodes 10 --no-render
    python evaluate.py runs/<run>/checkpoints --episodes 5 --stochastic   # score every checkpoint

When given a folder, every .zip in it is evaluated without a window and the
results are written to eval_results.csv in that folder.

Press q in the game window to stop early.
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np
from stable_baselines3 import PPO

from mario_rl.env import make_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Mario PPO agent")
    p.add_argument("model", help="path to a saved .zip model, or a folder of them")
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--actions", choices=["right_only", "simple", "complex"], default="simple")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely one")
    p.add_argument("--no-render", action="store_true", help="skip the game window (faster)")
    p.add_argument("--scale", type=int, default=3, help="window size multiplier")
    p.add_argument("--fps", type=int, default=30, help="playback speed in agent steps per second")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def run_episodes(model, env, args, render: bool, verbose: bool = True):
    """Play `args.episodes` episodes. Returns (clears, furthest x_pos per episode)."""
    game = env.venv.envs[0]
    clears, distances = 0, []
    obs = env.reset()
    quit_requested = False
    for ep in range(1, args.episodes + 1):
        done, steps, info = False, 0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, dones, infos = env.step(action)
            done, info = dones[0], infos[0]
            steps += 1
            if render:
                frame = cv2.cvtColor(game.render(), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Super Mario PPO", frame)
                if cv2.waitKey(max(1, 1000 // args.fps)) & 0xFF == ord("q"):
                    quit_requested = True
                    break
        if quit_requested:
            break
        cleared = bool(info.get("flag_get", False))
        clears += cleared
        distances.append(info.get("max_x_pos", 0))
        if verbose:
            print(f"episode {ep}: {'CLEARED' if cleared else 'failed'} | furthest x_pos {distances[-1]} | {steps} steps")
    return clears, distances


def evaluate_folder(folder, env, args):
    paths = sorted(glob.glob(os.path.join(folder, "*.zip")))
    if not paths:
        raise SystemExit(f"no .zip models found in {folder}")
    print(f"evaluating {len(paths)} models, {args.episodes} episodes each "
          f"({'stochastic' if args.stochastic else 'deterministic'} actions)\n")
    print(f"{'model':<45} {'clears':>8} {'mean x':>8} {'best x':>8}")
    rows = []
    for path in paths:
        model = PPO.load(path, device="cpu")
        clears, distances = run_episodes(model, env, args, render=False, verbose=False)
        name = os.path.basename(path)
        rows.append({"model": name, "clears": clears, "episodes": len(distances),
                     "clear_rate": clears / len(distances), "mean_x_pos": round(float(np.mean(distances))),
                     "best_x_pos": int(max(distances))})
        print(f"{name:<45} {clears:>4}/{len(distances):<3} {rows[-1]['mean_x_pos']:>8} {rows[-1]['best_x_pos']:>8}")

    out = os.path.join(folder, "eval_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda r: (r["clear_rate"], r["mean_x_pos"]))
    print(f"\nbest: {best['model']} (clear rate {best['clear_rate']:.0%}, mean x_pos {best['mean_x_pos']})")
    print(f"results saved to {out}")


def main():
    args = parse_args()
    env = make_vec_env(
        n_envs=1, seed=args.seed, subprocess=False,
        world=args.world, stage=args.stage, actions=args.actions,
    )
    if os.path.isdir(args.model):
        evaluate_folder(args.model, env, args)
    else:
        model = PPO.load(args.model, device="cpu")
        clears, distances = run_episodes(model, env, args, render=not args.no_render)
        if distances:
            print(f"\nclear rate {clears}/{len(distances)} | mean furthest x_pos {np.mean(distances):.0f} "
                  f"(flag is at roughly x_pos 3160 on World 1-1)")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
