"""Play a trained bot against the CPU or another bot, report its results and save replays.

Usage:
    python evaluate.py runs/bc_fox/best.pt --games 20 --n-envs 4            # fast, numbers
    python evaluate.py runs/bc_fox/best.pt --games 2 --n-envs 1 --watch     # normal speed, to watch
    python evaluate.py runs/bc_fox/best.pt --stage BATTLEFIELD --cpu-level 9 --replays runs/bc_fox/replays
    python evaluate.py runs/bc_fox/best.pt --ganondorf path/to/ganondorf_edge_v4.zip --games 2 --n-envs 1 --watch
    python evaluate.py runs/rl_fox/snapshots/fox_rl_003.00h.pt --versus runs/bc_fox/bc_fox_18h.pt --games 40
"""

import argparse
import os

import numpy as np
import torch

from smash_rl.agent import Agent
from smash_rl.ganondorf import GanondorfBot
from smash_rl.model import load_policy
from smash_rl.vec_env import make_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained bot against the CPU, the Ganondorf bot or another bot")
    p.add_argument("checkpoint", help="a .pt file from train_bc.py or train_rl.py")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=4, help="games played at once")
    p.add_argument("--opponent", default="FOX", help="CPU character, or RANDOM")
    p.add_argument("--ganondorf", default=None, metavar="MODEL_ZIP",
                   help="play against the Ganondorf bot (ganondorf_agent's model .zip, meta.json next to it) "
                        "instead of the CPU")
    p.add_argument("--versus", default=None, metavar="CHECKPOINT",
                   help="play against another trained bot (Fox, port 2, same delay) instead of the CPU")
    p.add_argument("--cpu-level", type=int, default=9)
    p.add_argument("--stage", default="FINAL_DESTINATION")
    p.add_argument("--delay", type=int, default=18, help="must match training")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (1 = as trained)")
    p.add_argument("--watch", action="store_true", help="run at normal speed (60 fps)")
    p.add_argument("--replays", default=None, help="directory for Slippi replays of the games")
    p.add_argument("--base-port", type=int, default=51600)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(load_policy(args.checkpoint, device), args.n_envs, device, args.temperature)
    bot, rival, opponent, against = None, None, args.opponent, f"CPU level {args.cpu_level} {args.opponent}"
    if args.ganondorf and args.versus:
        raise SystemExit("choose --ganondorf or --versus, not both")
    if args.ganondorf:
        bot = GanondorfBot.load(args.ganondorf)
        against = f"the {bot.character} bot"
        if bot.stage.name != args.stage.upper():
            print(f"note: the {bot.character} bot was trained on {bot.stage.name} only")
    if args.versus:
        # It sees the game from port 2: players swapped, so it is always "player 0" to itself.
        rival = Agent(load_policy(args.versus, device), args.n_envs, device, args.temperature)
        opponent, against = "POLICY", os.path.basename(args.versus)
    env = make_vec_env(n_envs=args.n_envs, base_port=args.base_port, opponent=opponent,
                       cpu_level=args.cpu_level, stages=(args.stage,), delay=args.delay,
                       emulation_speed=1.0 if args.watch else 0.0, opponent_bot=bot,
                       save_replays=os.path.abspath(args.replays) if args.replays else None)
    results = []
    try:
        obs, _ = env.reset()
        while len(results) < args.games:
            actions = agent.act(obs["players"], obs["stage"])
            if rival is not None:
                theirs = rival.act(obs["players"][:, ::-1], obs["stage"])
                actions = [np.stack(pair) for pair in zip(actions, theirs)]
            obs, _, dones, infos = env.step(actions)
            for i in np.flatnonzero(dones):
                info = infos[i]
                results.append(info)
                agent.reset(i)
                if rival is not None:
                    rival.reset(i)
                minutes = max(info["frames"], 1) / 3600
                print(f"game {len(results)}: {'WON ' if info['won'] else 'lost'} vs {against} | "
                      f"stocks {info['stock_diff']:+d} | kills {info['kills']} deaths {info['deaths']} | "
                      f"damage/min dealt {info['damage_dealt'] / minutes:.0f} taken {info['damage_taken'] / minutes:.0f}",
                      flush=True)
    finally:
        env.close()
    results = results[:args.games]
    print(f"\nwin rate {np.mean([r['won'] for r in results]):.0%} over {len(results)} games vs {against} "
          f"on {args.stage} | mean stock diff "
          f"{np.mean([r['stock_diff'] for r in results]):+.2f}")


if __name__ == "__main__":
    main()
