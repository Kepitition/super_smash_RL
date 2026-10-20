"""Reinforcement learning (PPO) from the imitation policy, kept close to it by a KL penalty.

Usage:
    python train_rl.py --run-dir runs/rl_fox --ganondorf path/to/ganondorf_edge_v4.zip --max-hours 10
    python train_rl.py --run-dir runs/rl_fox --ganondorf path/to/ganondorf_edge_v4.zip --resume --max-hours 10
    python train_rl.py --run-dir runs/rl_smoke --n-envs 3 --ganondorf-envs 0 --cpu-envs 1 --max-hours 0.05

Our Fox (port 1) plays n games at once: most against a past version of itself (self-play),
some against the Ganondorf bot and some against the level 9 CPU. Every `rollout` frames it
learns from what happened (smash_rl/ppo.py); rewards are smash_rl/reward.py. The KL
penalty to the imitation policy keeps it playing like a person.

Saved in run-dir: last.pt (resumable), snapshots/fox_rl_<hours>h.pt (play or evaluate these
with play.py / evaluate.py; they are also the self-play opponents) and TensorBoard logs:
tensorboard --logdir runs
"""

import argparse
import collections
import glob
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from smash_rl.agent import Agent
from smash_rl.env import make_env
from smash_rl.ganondorf import GanondorfBot
from smash_rl.model import load_policy
from smash_rl.ppo import Learner, Rollout, ValueNet
from smash_rl.vec_env import LocalVecEnv, SubprocVecEnv

GROUPS = ("self", "ganondorf", "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="PPO self-play from the imitation policy, with a KL leash to it")
    p.add_argument("--run-dir", required=True, help="checkpoints, snapshots and TensorBoard logs go here")
    p.add_argument("--resume", action="store_true", help="continue from run-dir/last.pt")
    p.add_argument("--max-hours", type=float, default=10.0, help="stop (and save) after this long")
    p.add_argument("--init", default="runs/bc_fox/bc_fox_18h.pt", help="imitation checkpoint to start from")
    p.add_argument("--teacher", default=None, help="imitation checkpoint for the KL penalty (default: --init)")
    # games
    p.add_argument("--n-envs", type=int, default=10, help="games played at once")
    p.add_argument("--ganondorf-envs", type=int, default=2, help="of which against the Ganondorf bot")
    p.add_argument("--cpu-envs", type=int, default=2, help="of which against the CPU (the rest: self-play)")
    p.add_argument("--ganondorf", default=None, metavar="MODEL_ZIP",
                   help="the Ganondorf bot's model .zip (meta.json next to it); needed for --ganondorf-envs")
    p.add_argument("--cpu-character", default="FOX")
    p.add_argument("--cpu-level", type=int, default=9)
    p.add_argument("--stage", default="FINAL_DESTINATION")
    p.add_argument("--delay", type=int, default=18, help="reaction delay in frames, must match the imitation policy")
    p.add_argument("--base-port", type=int, default=51620, help="Slippi port of the first game")
    # self-play opponents
    p.add_argument("--latest-prob", type=float, default=0.5,
                   help="chance the self-play opponent is the current policy (else a random earlier snapshot)")
    p.add_argument("--opponent-minutes", type=float, default=5.0, help="pick a new self-play opponent this often")
    p.add_argument("--snapshot-minutes", type=float, default=20.0, help="save a snapshot (and opponent) this often")
    # PPO
    p.add_argument("--rollout", type=int, default=512, help="frames per game between updates")
    p.add_argument("--chunk", type=int, default=128, help="frames per training sequence")
    p.add_argument("--epochs", type=int, default=2, help="passes over each rollout")
    p.add_argument("--minibatches", type=int, default=4, help="gradient steps per pass")
    p.add_argument("--lr", type=float, default=3e-5, help="policy learning rate")
    p.add_argument("--value-lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.997, help="discount per frame (0.997: half-life ~4 s)")
    p.add_argument("--lam", type=float, default=0.98, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO ratio clip")
    p.add_argument("--kl-coef", type=float, default=0.5,
                   help="weight of KL(policy || imitation policy) per frame: higher stays more human-like")
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--value-warmup", type=int, default=30, help="first updates train only the value network")
    # reward
    p.add_argument("--kill-reward", type=float, default=1.0, help="per stock taken (minus per stock lost)")
    p.add_argument("--damage-reward", type=float, default=0.01, help="per percent dealt (minus per percent taken)")
    p.add_argument("--ledge-penalty", type=float, default=0.02, help="per ledge grab")
    p.add_argument("--approach-reward", type=float, default=0.0002, help="per unit moved towards the opponent")
    p.add_argument("--stall-penalty", type=float, default=0.001, help="per frame offstage beyond --stall-seconds")
    p.add_argument("--stall-seconds", type=float, default=3.0)
    # logging
    p.add_argument("--log-every", type=int, default=10, help="updates between log lines")
    p.add_argument("--save-minutes", type=float, default=10.0, help="write last.pt this often")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_envs(args, groups):
    reward = dict(kill=args.kill_reward, damage=args.damage_reward, ledge_grab=args.ledge_penalty,
                  approach=args.approach_reward, stall=args.stall_penalty,
                  stall_frames=int(args.stall_seconds * 60))
    common = dict(stages=(args.stage,), delay=args.delay, reward=reward, cpu_level=args.cpu_level)
    bot = GanondorfBot.load(args.ganondorf) if "ganondorf" in groups else None
    thunks = []
    for i, group in enumerate(groups):
        kwargs = dict(common)
        if group == "self":
            kwargs["opponent"] = "POLICY"
        elif group == "ganondorf":
            kwargs["opponent_bot"] = bot
        else:
            kwargs["opponent"] = args.cpu_character
        thunks.append(make_env(i, args.base_port, **kwargs))
    return SubprocVecEnv(thunks) if len(thunks) > 1 else LocalVecEnv(thunks)


def weights_of(path: str, device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint.get("model", checkpoint)["state_dict"]


def main():
    args = parse_args()
    groups = (["self"] * (args.n_envs - args.ganondorf_envs - args.cpu_envs)
              + ["ganondorf"] * args.ganondorf_envs + ["cpu"] * args.cpu_envs)
    if min(args.n_envs - args.ganondorf_envs - args.cpu_envs, args.ganondorf_envs, args.cpu_envs) < 0:
        raise SystemExit("--ganondorf-envs + --cpu-envs can't exceed --n-envs")
    if args.ganondorf_envs and not args.ganondorf:
        raise SystemExit("--ganondorf-envs needs --ganondorf path/to/ganondorf_edge_v4.zip (or --ganondorf-envs 0)")
    n, n_self = len(groups), groups.count("self")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_dir = os.path.join(args.run_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    last_path = os.path.join(args.run_dir, "last.pt")

    teacher = load_policy(args.teacher or args.init, device).eval()
    policy = load_policy(args.init, device).eval()
    value = ValueNet.from_policy(policy).to(device)
    learner = Learner(policy, teacher, value, n, device, lr=args.lr, value_lr=args.value_lr, gamma=args.gamma,
                      lam=args.lam, clip=args.clip, kl_coef=args.kl_coef, entropy_coef=args.entropy_coef,
                      epochs=args.epochs, minibatches=args.minibatches, seed=args.seed)
    progress = {"updates": 0, "frames": 0, "hours": 0.0}
    if args.resume:
        saved = torch.load(last_path, map_location=device, weights_only=False)
        policy.load_state_dict(saved["model"]["state_dict"])
        value.load_state_dict(saved["value"]["state_dict"])
        learner.load_state_dict(saved["learner"])
        progress = saved["progress"]
        print(f"resumed after {progress['updates']} updates, {progress['frames'] / 1e6:.1f}M frames, "
              f"{progress['hours']:.2f} h")
    # Self-play opponents: the imitation policy and every snapshot so far.
    pool = [args.init] + sorted(glob.glob(os.path.join(snapshot_dir, "fox_rl_*.pt")))
    opponent = load_policy(args.init, device).eval() if n_self else None

    def save_last():
        torch.save({"model": policy.checkpoint(), "value": value.checkpoint(), "learner": learner.state_dict(),
                    "progress": progress, "args": vars(args)}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

    def save_snapshot():
        path = os.path.join(snapshot_dir, f"fox_rl_{progress['hours']:06.2f}h.pt")
        torch.save({"model": policy.checkpoint(), "progress": dict(progress)}, path)
        pool.append(path)
        return path

    def pick_opponent() -> str:
        if rng.random() < args.latest_prob:
            opponent.load_state_dict(policy.state_dict())  # copies in place: the CUDA graph stays valid
            return "latest"
        path = pool[rng.integers(len(pool))]
        opponent.load_state_dict(weights_of(path, device))
        return os.path.basename(path)

    print(f"{n} games: {n_self} self-play, {args.ganondorf_envs} vs Ganondorf, {args.cpu_envs} vs CPU level "
          f"{args.cpu_level} {args.cpu_character} | starting the Dolphins...", flush=True)
    env = make_envs(args, groups)
    writer = SummaryWriter(args.run_dir)
    recent = {g: collections.deque(maxlen=50) for g in GROUPS}
    session_start = time.time()
    last_save = last_snapshot = last_swap = time.time()
    hours_at_start = progress["hours"]
    try:
        obs, _ = env.reset()
        agent = Agent(policy, n, device)
        opponent_agent = Agent(opponent, n_self, device) if n_self else None
        opponent_name = pick_opponent() if n_self else None
        first = np.ones(n, dtype=bool)
        print(f"playing | self-play opponent: {opponent_name}", flush=True)
        window_start, window_frames = time.time(), 0
        timing = collections.Counter()  # seconds spent acting, in the games and learning
        while True:
            rollout = Rollout(n, args.rollout, args.chunk)
            for t in range(args.rollout):
                if t % args.chunk == 0:
                    rollout.chunk_states.append(agent.state_copy())
                rollout.add_frame(obs["players"], obs["stage"], first, agent.prev)
                tick = time.perf_counter()
                actions = agent.act(obs["players"], obs["stage"])
                logp = agent.logp
                step_actions = list(actions)
                if n_self:
                    theirs = opponent_agent.act(obs["players"][:n_self, ::-1], obs["stage"][:n_self])
                    for i in range(n_self):
                        step_actions[i] = np.stack([actions[i], theirs[i]])
                tock = time.perf_counter()
                obs, rewards, dones, infos = env.step(step_actions)
                timing["act"] += tock - tick
                timing["games"] += time.perf_counter() - tock
                rollout.add_step(actions, logp, rewards, dones)
                first = dones.copy()
                for i in np.flatnonzero(dones):
                    agent.reset(i)
                    if i < n_self:
                        opponent_agent.reset(i)
                    info = infos[i]
                    if not info.get("crashed"):
                        recent[groups[i]].append(info)
            rollout.add_frame(obs["players"], obs["stage"], first, agent.prev)

            warmup = progress["updates"] < args.value_warmup
            tick = time.perf_counter()
            stats = learner.update(rollout, train_policy=not warmup)
            timing["learn"] += time.perf_counter() - tick
            progress["updates"] += 1
            progress["frames"] += n * args.rollout
            window_frames += n * args.rollout
            now = time.time()
            progress["hours"] = hours_at_start + (now - session_start) / 3600
            step = progress["frames"]
            for k, v in stats.items():
                writer.add_scalar(f"train/{k}", v, step)
            if progress["updates"] % args.log_every == 0:
                fps = window_frames / (now - window_start)
                shares = {k: v / (now - window_start) for k, v in timing.items()}
                window_start, window_frames = now, 0
                timing.clear()
                writer.add_scalar("speed/frames_per_second", fps, step)
                for k, v in shares.items():
                    writer.add_scalar(f"speed/share_{k}", v, step)
                results = []
                for g in GROUPS:
                    games = recent[g]
                    if not games:
                        continue
                    win = float(np.mean([r["won"] for r in games]))
                    writer.add_scalar(f"games/{g}_win_rate", win, step)
                    writer.add_scalar(f"games/{g}_stock_diff", float(np.mean([r["stock_diff"] for r in games])), step)
                    results.append(f"{g} {win:.0%} of {len(games)}")
                print(f"{progress['hours']:.2f} h | update {progress['updates']}{' (value warm-up)' if warmup else ''} | "
                      f"{step / 1e6:.1f}M frames, {fps:.0f}/s (time: games {shares.get('games', 0):.0%}, "
                      f"acting {shares.get('act', 0):.0%}, learning {shares.get('learn', 0):.0%}) | KL to human {stats['teacher_kl']:.3f} | "
                      f"clipped {stats['clip_frac']:.0%} | value EV {stats['explained_variance']:.2f} | "
                      f"wins (last 50): {', '.join(results) or '-'}", flush=True)
            if n_self and now - last_swap >= args.opponent_minutes * 60:
                opponent_name = pick_opponent()
                last_swap = now
                print(f"self-play opponent: {opponent_name}", flush=True)
            if now - last_snapshot >= args.snapshot_minutes * 60:
                print(f"snapshot: {save_snapshot()}", flush=True)
                last_snapshot = now
            if now - last_save >= args.save_minutes * 60:
                save_last()
                last_save = now
            if now - session_start >= args.max_hours * 3600:
                print(f"stopping after {args.max_hours} h", flush=True)
                break
    except KeyboardInterrupt:
        print("\nstopping (Ctrl+C)", flush=True)
    finally:
        if progress["updates"]:
            save_last()
            print(f"saved {last_path} and snapshot {save_snapshot()}", flush=True)
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
