"""Watch a trained PPO agent play Melee against the CPU and report its win rate.

Usage:
    python evaluate.py runs/<run>/final_model.zip                       # 3 games vs a level 9 CPU, normal speed
    python evaluate.py runs/<run>/final_model.zip --games 10 --cpu-level 5 --fast
    python evaluate.py runs/<run>/checkpoints --games 3 --headless      # score every checkpoint
    python evaluate.py runs/<run>/final_model.zip --opponent RANDOM     # a random CPU character every game

The game plays in the Dolphin window. By default it runs at normal speed (60 fps)
so you can watch; --fast runs it as fast as the emulator can go, and --headless
(Linux) runs it with no window at all, which is fastest.

When given a folder, every .zip in it is evaluated and the results are written
to eval_results.csv in that folder.
"""

import argparse
import csv
import glob
import os

import numpy as np
from stable_baselines3 import PPO

from smash_rl.env import SmashEnv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Melee PPO agent")
    p.add_argument("model", help="path to a saved .zip model, or a folder of them")
    p.add_argument("--character", default="FOX")
    p.add_argument("--opponent", default="FOX", help="CPU character, or RANDOM for a new random character every game")
    p.add_argument("--stage", default="FINAL_DESTINATION")
    p.add_argument("--cpu-level", type=int, default=9)
    p.add_argument("--frame-skip", type=int, default=3, help="must match training")
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely one")
    p.add_argument("--fast", action="store_true", help="run the emulator unthrottled instead of at 60 fps")
    p.add_argument("--headless", action="store_true", help="no window and fast-forward mode (Linux only, fastest)")
    p.add_argument("--port", type=int, default=51500)
    return p.parse_args()


def play_games(model, env, games: int, deterministic: bool, verbose: bool = True):
    """Play `games` full games. Returns a list of per-game info dicts."""
    results = []
    for game in range(1, games + 1):
        obs, info = env.reset()
        done, steps = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        results.append(info)
        if verbose:
            print(f"game {game} vs {info['opponent']}: {'WON ' if info.get('won') else 'lost'} | "
                  f"stock diff {info.get('stock_diff', 0):+d} | "
                  f"kills {info['kills']} deaths {info['deaths']} | damage dealt {info['damage_dealt']:.0f}% "
                  f"taken {info['damage_taken']:.0f}% | {steps} steps")
    return results


def summarize(results):
    return {
        "games": len(results),
        "win_rate": float(np.mean([r.get("won", 0) for r in results])),
        "mean_stock_diff": float(np.mean([r.get("stock_diff", 0) for r in results])),
        "mean_kills": float(np.mean([r["kills"] for r in results])),
        "mean_deaths": float(np.mean([r["deaths"] for r in results])),
    }


def evaluate_folder(folder, env, args):
    paths = sorted(glob.glob(os.path.join(folder, "*.zip")))
    if not paths:
        raise SystemExit(f"no .zip models found in {folder}")
    print(f"evaluating {len(paths)} models, {args.games} games each vs CPU level {args.cpu_level}\n")
    print(f"{'model':<45} {'win rate':>9} {'stock diff':>11} {'kills':>6} {'deaths':>7}")
    rows = []
    for path in paths:
        model = PPO.load(path, device="cpu")
        s = summarize(play_games(model, env, args.games, not args.stochastic, verbose=False))
        rows.append({"model": os.path.basename(path), **s})
        print(f"{rows[-1]['model']:<45} {s['win_rate']:>9.0%} {s['mean_stock_diff']:>+11.2f} "
              f"{s['mean_kills']:>6.1f} {s['mean_deaths']:>7.1f}")

    out = os.path.join(folder, "eval_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda r: (r["win_rate"], r["mean_stock_diff"]))
    print(f"\nbest: {best['model']} (win rate {best['win_rate']:.0%}, stock diff {best['mean_stock_diff']:+.2f})")
    print(f"results saved to {out}")


def main():
    args = parse_args()
    env = SmashEnv(
        character=args.character, opponent=args.opponent, stage=args.stage,
        cpu_level=args.cpu_level, frame_skip=args.frame_skip, port=args.port,
        emulation_speed=0.0 if args.fast or args.headless else 1.0, headless=args.headless,
    )
    try:
        if os.path.isdir(args.model):
            evaluate_folder(args.model, env, args)
        else:
            model = PPO.load(args.model, device="cpu")
            s = summarize(play_games(model, env, args.games, not args.stochastic))
            print(f"\nwin rate {s['win_rate']:.0%} over {s['games']} games vs CPU level {args.cpu_level} | "
                  f"mean stock diff {s['mean_stock_diff']:+.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
