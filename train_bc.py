"""Behavior cloning: learn Fox's controller from human replays.

Usage:
    python train_bc.py --run-dir runs/bc_fox --max-hours 10              # one night
    python train_bc.py --run-dir runs/bc_fox --resume --max-hours 10     # continue from last.pt
    python train_bc.py --run-dir runs/bc_smoke --max-hours 0.05 --batch-size 64 --eval-minutes 1

Monitor with: tensorboard --logdir runs
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from smash_rl.data import Prefetcher, UnrollBatcher, load_index, to_torch, trajectory_list
from smash_rl.model import Policy

HEADS = ("buttons", "stick_x", "stick_y", "cstick")


def parse_args():
    p = argparse.ArgumentParser(description="Imitation learning on parsed replays")
    p.add_argument("--data-dir", default=r"D:\smash_data", help="output of prepare_data.py")
    p.add_argument("--run-dir", required=True, help="checkpoints and TensorBoard logs go here")
    p.add_argument("--resume", action="store_true", help="continue from run-dir/last.pt")
    p.add_argument("--max-hours", type=float, default=10.0, help="stop (and save) after this long")
    p.add_argument("--delay", type=int, default=18, help="frames between the observed frame and the label")
    p.add_argument("--batch-size", type=int, default=512, help="parallel trajectories per batch")
    p.add_argument("--unroll", type=int, default=80, help="frames per chunk")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--eval-minutes", type=float, default=10.0, help="validate and checkpoint this often")
    p.add_argument("--val-batches", type=int, default=50, help="chunks per validation (fixed set)")
    p.add_argument("--log-every", type=int, default=100, help="steps between training log lines")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def validate(model, trajectories, args, device):
    model.eval()
    batcher = UnrollBatcher(trajectories, args.batch_size, args.unroll, args.delay, seed=12345)
    state = model.initial_state(args.batch_size, device)
    losses, accs = [], []
    for _ in range(args.val_batches):
        batch = to_torch(batcher.next_batch(), device)
        with torch.autocast("cuda", dtype=torch.float16):
            out, state = model(batch, state, batch["reset"])
            loss, _, head_accs = model.loss(out, batch["target"])
        losses.append(loss.item())
        accs.append([a.item() for a in head_accs])
    model.train()
    return float(np.mean(losses)), np.mean(accs, axis=0)


def save(path, model, opt, scaler, step, hours, best_val, args):
    torch.save({"model": model.checkpoint(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "hours": hours, "best_val": best_val, "args": vars(args)}, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    index = load_index(args.data_dir)
    train, val = trajectory_list(index, "train", args.data_dir), trajectory_list(index, "val", args.data_dir)
    print(f"{len(index)} games: {len(train)} train and {len(val)} val trajectories")

    model = Policy(width=args.width, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    step, hours, best_val = 0, 0.0, float("inf")
    last_path, best_path = os.path.join(args.run_dir, "last.pt"), os.path.join(args.run_dir, "best.pt")
    if args.resume:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"]["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, hours, best_val = ckpt["step"], ckpt["hours"], ckpt["best_val"]
        print(f"resumed at step {step} ({hours:.1f} h trained, best val {best_val:.4f})")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    writer = SummaryWriter(args.run_dir)
    batches = Prefetcher(UnrollBatcher(train, args.batch_size, args.unroll, args.delay, seed=args.seed + step))
    state = model.initial_state(args.batch_size, device)
    start = last_eval = time.time()
    while True:
        batch = to_torch(batches.next(), device)
        with torch.autocast("cuda", dtype=torch.float16):
            out, state = model(batch, state, batch["reset"])
            loss, head_losses, head_accs = model.loss(out, batch["target"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        state = [(h.detach(), c.detach()) for h, c in state]
        step += 1

        if step % args.log_every == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            for name, l, a in zip(HEADS, head_losses, head_accs):
                writer.add_scalar(f"train/{name}_loss", l.item(), step)
                writer.add_scalar(f"train/{name}_acc", a.item(), step)
            print(f"step {step} loss {loss.item():.4f} | acc " +
                  " ".join(f"{n} {a.item():.3f}" for n, a in zip(HEADS, head_accs)), flush=True)

        elapsed = (time.time() - start) / 3600
        done = elapsed >= args.max_hours
        if done or time.time() - last_eval >= args.eval_minutes * 60:
            last_eval = time.time()
            val_loss, val_accs = validate(model, val, args, device)
            writer.add_scalar("val/loss", val_loss, step)
            for name, a in zip(HEADS, val_accs):
                writer.add_scalar(f"val/{name}_acc", a, step)
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            save(last_path, model, opt, scaler, step, hours + elapsed, best_val, args)
            if improved:
                save(best_path, model, opt, scaler, step, hours + elapsed, best_val, args)
            print(f"val loss {val_loss:.4f}{' (best)' if improved else ''} | acc " +
                  " ".join(f"{n} {a:.3f}" for n, a in zip(HEADS, val_accs)), flush=True)
        if done:
            print(f"stopped after {elapsed:.2f} h this session ({hours + elapsed:.1f} h in total)")
            break


if __name__ == "__main__":
    main()
