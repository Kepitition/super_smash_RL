"""Figures and numbers for the poster, from the training logs and evaluation games.

Run from the repository root:
    python poster/make_figures.py                     # redraw from the existing results
    python poster/make_figures.py --evaluate runs/rl_fox/snapshots/fox_rl_013.00h.pt --ganondorf C:\\path\\to\\ganondorf_edge_v4.zip

--evaluate first plays the evaluation games of a new RL snapshot (~40 minutes, Dolphin
windows open): against the level 9 CPU, the Ganondorf bot and the imitation policy. Then
build the PDF with poster/build.ps1.

Writes poster/fig/*.pdf (vector, for LaTeX), *.png previews, and fig/results.tex: every
number the poster text shows. Inputs (not in git): runs/bc_fox and runs/rl_fox TensorBoard
logs and the evaluation logs and replays in runs/poster. The imitation policy against the
CPU is runs/bc_fox/eval18_cpu9_fd. Human play style comes from Fox dittos in the replay
dataset's validation split, cached in fig/human_style.json.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
OUT = os.path.join("poster", "fig")
RESULTS = os.path.join("runs", "poster")
BC_CPU = os.path.join("runs", "bc_fox", "eval18_cpu9_fd")
RL, BC, HUMAN, OPPONENT, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#52514e", "#a3a29c", "#0b0b0b", "#52514e", "#e4e4e2"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 12, "axes.edgecolor": "#8a8985",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2.5, "pdf.fonttype": 42,
    "legend.frameon": False,
})
# Screenshots of the bot's combos, from poster/capture_combos.py (3-hour snapshot vs CPU 9):
# (combo folder, game frame, crop box left, top, right, bottom in pixels of the 1280x960 window).
FRAMES = os.path.join("poster", "frames")
HERO = ("combo2", 3421, (96, 175, 1136, 695))  # the shine at the end of forward-air > drill > shine
STRIP = ("combo5", [(7264, (350, 215, 910, 635)), (7307, (400, 210, 960, 630)),
                    (7340, (310, 330, 870, 750)), (7375, (340, 260, 900, 680))])

# Fox action states (libmelee ids) whose first frame counts as one use of the move.
MOVES = {
    "jab": [44], "dash attack": [50], "ftilt": [51, 52, 53, 54, 55], "uptilt": [56], "dtilt": [57],
    "fsmash": [58, 59, 60, 61, 62], "upsmash": [63], "dsmash": [64],
    "nair": [65], "fair": [66], "bair": [67], "uair": [68], "dair": [69],
    "grab": [212, 214], "throw": [219, 220, 221, 222],
    "laser": [341, 344], "shine": [360, 365], "side-B": [349, 350], "up-B": [353, 354],
}
LANDING, AIRDODGE = 43, 236
COMBO_GAP = 45  # frames: the next hit within this keeps the combo going
SELF_DESTRUCT_GAP = 240  # frames: a death without being hit this long before
STYLE_KEYS = {"Attacks / min": "moves_per_min", "Move variety (bits)": "variety_bits",
              "Wavedashes / min": "wavedash_per_min", "Shines / min": "shine_per_min",
              "Hits per combo": "hits_per_combo"}


def parse_args():
    p = argparse.ArgumentParser(description="Poster figures and numbers from the logs and evaluation games")
    p.add_argument("--evaluate", metavar="SNAPSHOT", default=None,
                   help="first play the evaluation games of this RL snapshot (~40 min)")
    p.add_argument("--ganondorf", metavar="MODEL_ZIP", default=None, help="the Ganondorf bot (needed with --evaluate)")
    p.add_argument("--imitation", default="runs/bc_fox/bc_fox_18h.pt", help="the imitation policy")
    p.add_argument("--games", type=int, default=20, help="games against the CPU and against the Ganondorf bot")
    p.add_argument("--versus-games", type=int, default=40, help="games against the imitation policy")
    p.add_argument("--n-envs", type=int, default=5, help="games at once per evaluation (two run in parallel)")
    p.add_argument("--data-dir", default=r"D:\smash_data", help="prepare_data.py output, for the human style")
    return p.parse_args()


# ---------------------------------------------------------------- evaluation games

def evaluate(args):
    from smash_rl.model import load_policy  # noqa: F401  (fails early if the snapshot can't load)

    if not args.ganondorf:
        raise SystemExit("--evaluate needs --ganondorf path/to/ganondorf_edge_v4.zip")
    os.makedirs(RESULTS, exist_ok=True)
    snap = args.evaluate
    jobs = [  # (log name, evaluate.py arguments); two run at once on different ports
        ("rl_vs_bc", [snap, "--versus", args.imitation, "--games", str(args.versus_games),
                      "--replays", os.path.join(RESULTS, "rl_vs_bc")]),
        ("rl_cpu9", [snap, "--games", str(args.games), "--replays", os.path.join(RESULTS, "rl_cpu9")]),
        ("rl_ganondorf", [snap, "--ganondorf", args.ganondorf, "--games", str(args.games)]),
    ]
    if not os.path.isfile(os.path.join(RESULTS, "bc_ganondorf.log")):
        jobs.append(("bc_ganondorf", [args.imitation, "--ganondorf", args.ganondorf, "--games", str(args.games)]))
    for name, _ in jobs:  # new replays replace the old ones (the combo is cached)
        for old in glob.glob(os.path.join(RESULTS, name, "**", "*.slp"), recursive=True):
            os.remove(old)
    running, ports = [], [51600, 51640]
    for name, extra in jobs:
        if len(running) == 2:
            proc, port = running.pop(0)
            proc.wait()
            ports.append(port)
        port = ports.pop(0)
        log = open(os.path.join(RESULTS, name + ".log"), "w")
        cmd = [sys.executable, "evaluate.py", *extra, "--n-envs", str(args.n_envs), "--base-port", str(port)]
        print("playing:", name, flush=True)
        running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), port))
    for proc, _ in running:
        proc.wait()
    with open(os.path.join(RESULTS, "snapshot.txt"), "w") as f:
        f.write(snap)


def eval_result(log: str) -> dict:
    with open(log, encoding="utf-8", errors="replace") as f:
        m = re.search(r"win rate (\d+)% over (\d+) games .*mean stock diff ([+-]?[\d.]+)", f.read())
    if not m:
        raise SystemExit(f"{log} has no final result line: did the evaluation finish?")
    return {"win": int(m.group(1)), "games": int(m.group(2)), "lead": float(m.group(3))}


# ---------------------------------------------------------------- play style

def starts(action, ids):
    a = np.isin(action, ids)
    return int((a[1:] & ~a[:-1]).sum())


def style_of(players) -> dict:
    """Style of player 0 in raw frames (T, 2) [player, opponent], after the countdown."""
    p, o = players[123:, 0], players[123:, 1]
    minutes = len(p) / 3600
    moves = {k: starts(p["action"], v) for k, v in MOVES.items()}
    total = max(sum(moves.values()), 1)
    probs = np.array([v / total for v in moves.values() if v])
    landing = (p["action"][1:] == LANDING) & (p["action"][:-1] != LANDING)
    dodged = np.array([np.any(p["action"][max(0, i - 8):i + 1] == AIRDODGE) for i in range(len(p) - 1)])
    hit = np.flatnonzero((np.diff(o["percent"]) > 0) & (np.diff(o["stocks"].astype(int)) == 0)) + 1
    combos, cur = [], 1
    for a, b in zip(hit[:-1], hit[1:]):
        if b - a <= COMBO_GAP:
            cur += 1
        else:
            combos.append(cur)
            cur = 1
    if len(hit):
        combos.append(cur)
    deaths = np.flatnonzero(np.diff(p["stocks"].astype(int)) < 0) + 1
    was_hit = np.flatnonzero(np.diff(p["percent"]) > 0) + 1
    self_destructs = sum(1 for d in deaths if not np.any((was_hit < d) & (was_hit >= d - SELF_DESTRUCT_GAP)))
    return {"moves_per_min": total / minutes, "variety_bits": float(-(probs * np.log2(probs)).sum()),
            "wavedash_per_min": float((landing & dodged).sum()) / minutes, "shine_per_min": moves["shine"] / minutes,
            "hits_per_combo": float(np.mean(combos)) if combos else 0.0,
            "deaths": len(deaths), "self_destructs": self_destructs}


def replay_style(folder: str, port: int) -> dict:
    """Mean style of one port over a folder of replays, plus its L-cancel rate and deaths."""
    import peppi_py

    from smash_rl.state import from_peppi

    rows, lc = [], [0, 0]
    for replay in sorted(glob.glob(os.path.join(folder, "**", "*.slp"), recursive=True)):
        try:
            game = peppi_py.read_slippi(replay)
            players, _, _ = from_peppi(game)
        except Exception:
            continue  # an unfinished game
        rows.append(style_of(players if port == 0 else players[:, ::-1]))
        flags = np.asarray(game.frames.ports[port].leader.post.l_cancel)
        lc[0] += int((flags == 1).sum())
        lc[1] += int((flags == 2).sum())
    if not rows:
        raise SystemExit(f"no readable replays in {folder}")
    out = {k: float(np.mean([r[k] for r in rows])) for k in STYLE_KEYS.values()}
    out.update(games=len(rows), deaths=sum(r["deaths"] for r in rows),
               self_destructs=sum(r["self_destructs"] for r in rows), l_cancel=lc[0] / max(sum(lc), 1))
    return out


def human_style(data_dir: str) -> dict:
    cache = os.path.join(OUT, "human_style.json")
    if os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    from smash_rl.data import load_index

    index = [e for e in load_index(data_dir) if e["split"] == "val" and e["characters"] == [1, 1]]
    dittos = [e for e in index if e["stage"] == 0][:40]  # Fox dittos on Final Destination
    rows = []
    for e in dittos:
        with np.load(os.path.join(data_dir, e["path"])) as d:
            players = d["players"]
        for slot in e["slots"]:
            rows.append(style_of(players if slot == 0 else players[:, ::-1]))
    out = {k: float(np.mean([r[k] for r in rows])) for k in STYLE_KEYS.values()}
    out.update(players=len(rows), deaths=sum(r["deaths"] for r in rows),
               self_destructs=sum(r["self_destructs"] for r in rows))
    with open(cache, "w") as f:
        json.dump(out, f, indent=1)
    return out


# ---------------------------------------------------------------- training curves

def scalars(pattern: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    steps, values, walls = [], [], []
    for path in sorted(glob.glob(pattern)):
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                steps.append(e.step)
                values.append(e.value)
                walls.append(e.wall_time)
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(values)[order], np.array(walls)[order]


def training_hours(walls):
    """Hours of training at each point, leaving out the breaks between resumed sessions."""
    dt = np.diff(walls, prepend=walls[0])
    dt[dt > 600] = np.median(dt[dt <= 600]) if np.any(dt <= 600) else 0
    return np.cumsum(dt) / 3600


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)


def imitation_curve():
    steps, loss, _ = scalars("runs/bc_fox/events.*", "val/loss")
    hours = steps / 19_000  # ~19k updates per hour on the RTX 4060 laptop GPU
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(hours, loss, color=BC)
    ax.set_xlabel("training hours")
    ax.set_ylabel("validation loss")
    ax.set_xticks([0, 6, 12, 18])
    ax.annotate(f"{loss[-1]:.2f}", (hours[-1], loss[-1]), xytext=(0, 10), textcoords="offset points",
                ha="right", color=INK, fontsize=13)
    save(fig, "imitation_loss")


def rl_curves(bc_ganondorf: float, bc_cpu: float):
    panels = [  # tag, title, scale, imitation baseline, end-label format, y-limits, smoothing window
        ("games/ganondorf_win_rate", "Ganondorf win %", 100.0, bc_ganondorf, "{:.0f}%", (0, 115), 1),
        ("games/cpu_stock_diff", "CPU 9 stock lead", 1.0, bc_cpu, "{:+.1f}", (0, 4.6), 1),
        ("train/teacher_kl", "KL to human", 1.0, None, None, (0, 0.05), 25),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 1.8))
    for ax, (tag, title, scale, baseline, fmt, ylim, smooth) in zip(axes, panels):
        _, v, w = scalars("runs/rl_fox/events.*", tag)
        h = training_hours(w)
        if smooth > 1:
            v = np.convolve(v, np.ones(smooth) / smooth, mode="valid")
            h = h[smooth - 1:]
        keep = h >= 0.25  # the 50-game windows are nearly empty before that
        ax.plot(h[keep], v[keep] * scale, color=RL)
        end_h = float(np.ceil(h[-1]))
        if baseline is not None:
            ax.axhline(baseline, color=BC, linestyle=(0, (4, 3)), linewidth=2.2)
            ax.text(end_h * 0.98, baseline, "imitation", color=BC, ha="right", va="top", fontsize=12,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=4)
        ax.set_xlim(0, end_h)
        ax.set_ylim(*ylim)
        ax.set_xlabel("RL hours")
        if fmt:
            end = v[keep][-1] * scale
            ax.text(h[keep][-1], end + (ylim[1] - ylim[0]) * 0.04, fmt.format(end), ha="right", va="bottom",
                    color=INK, fontsize=13)
    axes[0].set_yticks([0, 50, 100])
    axes[0].set_yticklabels(["0%", "50%", "100%"])
    axes[2].set_yticks([0, 0.02, 0.04])
    fig.tight_layout(w_pad=0.6)
    save(fig, "rl_curves")


def style_figure(human, bc, rl):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.15), gridspec_kw={"width_ratios": [1.75, 1.0]})
    names = list(STYLE_KEYS)[::-1]
    y = np.arange(len(names))
    rel_bc = [100 * bc[STYLE_KEYS[n]] / human[STYLE_KEYS[n]] for n in names]
    rel_rl = [100 * rl[STYLE_KEYS[n]] / human[STYLE_KEYS[n]] for n in names]
    ax.axvline(100, color=HUMAN, linewidth=2.5, zorder=2)
    for i in y:
        ax.plot([rel_bc[i], rel_rl[i]], [i, i], color=GRID, linewidth=3, zorder=1)
    ax.scatter(rel_bc, y, s=110, color=BC, edgecolor="white", linewidth=2, zorder=3, label="Imitation")
    ax.scatter(rel_rl, y, s=110, color=RL, edgecolor="white", linewidth=2, zorder=3, label="+ RL")
    ax.plot([], [], color=HUMAN, linewidth=2.5, label="Humans")
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    ax.set_ylim(-0.6, len(names) - 0.4)
    low = min(75, 5 * np.floor(min(rel_bc + rel_rl) / 5) - 5)
    high = max(125, 5 * np.ceil(max(rel_bc + rel_rl) / 5) + 5)
    ax.set_xlim(low, high)
    ax.set_xticks([t for t in (60, 80, 100, 120, 140) if low <= t <= high])
    ax.set_xticklabels([f"{t}%" for t in ax.get_xticks()])
    ax.set_xlabel("relative to humans")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.75, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.0,
              fontsize=12, handlelength=1.2)

    shares = [100 * d["self_destructs"] / max(d["deaths"], 1) for d in (human, bc, rl)]
    bars = bx.bar(range(3), shares, color=[HUMAN, BC, RL], width=0.6, edgecolor="white", linewidth=2)
    for bar, share in zip(bars, shares):
        bx.text(bar.get_x() + bar.get_width() / 2, share + 0.6, f"{share:.0f}%", ha="center", va="bottom",
                color=INK, fontsize=13)
    top = max(22, 5 * np.ceil(max(shares) / 5) + 5)
    bx.set_xticks(range(3))
    bx.set_xticklabels(["Humans", "Imit.", "+ RL"], fontsize=12)
    bx.set_ylim(0, top)
    bx.set_yticks([0, 10, 20] if top <= 25 else [0, 10, 20, 30])
    bx.set_yticklabels([f"{t}%" for t in bx.get_yticks()])
    bx.grid(axis="x", visible=False)
    bx.set_title("Self-destructs", loc="left", fontsize=12.5, pad=4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "style")


def screenshot(combo: str, frame: int):
    """The captured window picture of `frame`, or the next frame whose capture isn't black."""
    from PIL import Image

    folder = os.path.join(FRAMES, combo)
    for f in range(frame, frame + 8):
        path = os.path.join(folder, f"frame{f:06d}.jpg")
        if os.path.isfile(path):
            image = Image.open(path).convert("RGB")
            if np.asarray(image).mean() > 8:  # a few captures catch the window mid-redraw
                return image
    raise SystemExit(f"no usable screenshot near frame {frame} in {folder}")


def screenshots():
    """fig/hero.jpg (one big moment), fig/combo_strip.jpg and fig/combo_strip.tex (its labels)."""
    from PIL import Image

    if not os.path.isdir(FRAMES):
        print(f"no {FRAMES}: keeping the existing screenshots (run poster/capture_combos.py)")
        return
    combo, frame, box = HERO
    screenshot(combo, frame).crop(box).save(os.path.join(OUT, "hero.jpg"), quality=92)
    combo, shots = STRIP
    with open(os.path.join(FRAMES, combo, "hits.json")) as f:
        hits = json.load(f)
    tiles = [screenshot(combo, frame).crop(box).resize((560, 420)) for frame, box in shots]
    gap = 12
    strip = Image.new("RGB", (len(tiles) * 560 + (len(tiles) - 1) * gap, 420), "white")
    for i, tile in enumerate(tiles):
        strip.paste(tile, (i * (560 + gap), 0))
    strip.save(os.path.join(OUT, "combo_strip.jpg"), quality=92)
    labels = []
    for i, (frame, _) in enumerate(shots):
        hit = max((h for h in hits if h["frame"] <= frame), key=lambda h: h["frame"])
        labels.append(f"\\textbf{{{i + 1}}} {hit['move']} {{\\color{{black!55}}{hit['percent']:.0f}\\%}}")
    with open(os.path.join(OUT, "combo_strip.tex"), "w") as f:
        f.write("% Written by poster/make_figures.py: labels under fig/combo_strip.jpg.\n")
        f.write("{\\small\\begin{tabularx}{\\linewidth}{" + "Z" * len(labels) + "}\n" + " & ".join(labels)
                + "\n\\end{tabularx}}\n")


# ---------------------------------------------------------------- numbers for the text

def write_numbers(results, human, bc, rl, versus_bc, versus_rl, rl_hours):
    def cell(r):
        return f"{r['win']}\\% \\;({r['lead']:+.1f})"

    def share(d):
        return f"{100 * d['self_destructs'] / max(d['deaths'], 1):.0f}\\%"

    _, kl, _ = scalars("runs/rl_fox/events.*", "train/teacher_kl")
    _, fps, _ = scalars("runs/rl_fox/events.*", "speed/frames_per_second")
    spread = max(abs(100 * d[k] / human[k] - 100) for d in (bc, rl)
                 for k in ("moves_per_min", "variety_bits", "wavedash_per_min", "shine_per_min"))
    lc = sorted([100 * bc["l_cancel"], 100 * rl["l_cancel"]])
    numbers = {
        "rlHours": f"{rl_hours:.0f}",
        "cpuGames": results["cpu_rl"]["games"], "ganonGames": results["ganon_rl"]["games"],
        "vsGames": results["versus"]["games"],
        "cpuBC": cell(results["cpu_bc"]), "cpuRL": cell(results["cpu_rl"]),
        "ganonBC": cell(results["ganon_bc"]), "ganonRL": cell(results["ganon_rl"]), "vsRL": cell(results["versus"]),
        "cpuRLwin": f"{results['cpu_rl']['win']}\\%", "ganonRLwin": f"{results['ganon_rl']['win']}\\%",
        "cpuBCwin": f"{results['cpu_bc']['win']}\\%", "ganonBCwin": f"{results['ganon_bc']['win']}\\%",
        "vsRLwin": f"{results['versus']['win']}\\%",
        "sdHuman": share(human), "sdBC": share(versus_bc), "sdRL": share(versus_rl),
        "comboBC": f"{bc['hits_per_combo']:.2f}", "comboRL": f"{rl['hits_per_combo']:.2f}",
        "klLevel": f"{np.mean(kl[-max(1, len(kl) // 10):]):.2f}",
        "lcancel": f"{lc[0]:.0f}--{lc[1]:.0f}\\%" if round(lc[0]) != round(lc[1]) else f"{lc[0]:.0f}\\%",
        "styleSpread": f"{int(np.ceil(spread))}\\%", "fps": f"{10 * round(np.median(fps) / 10):.0f}",
        "humanPlayers": human["players"],
    }
    with open(os.path.join(OUT, "results.tex"), "w") as f:
        f.write("% Written by poster/make_figures.py: numbers the poster text uses.\n")
        for name, value in numbers.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    return numbers


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.evaluate:
        evaluate(args)
    results = {
        "cpu_bc": eval_result(BC_CPU + ".log"), "cpu_rl": eval_result(os.path.join(RESULTS, "rl_cpu9.log")),
        "ganon_bc": eval_result(os.path.join(RESULTS, "bc_ganondorf.log")),
        "ganon_rl": eval_result(os.path.join(RESULTS, "rl_ganondorf.log")),
        "versus": eval_result(os.path.join(RESULTS, "rl_vs_bc.log")),
    }
    with open(os.path.join(RESULTS, "snapshot.txt")) as f:
        snapshot = f.read().strip()
    import torch

    rl_hours = torch.load(snapshot, map_location="cpu", weights_only=False)["progress"]["hours"]
    human = human_style(args.data_dir)
    bc, rl = replay_style(BC_CPU, 0), replay_style(os.path.join(RESULTS, "rl_cpu9"), 0)
    versus_rl = replay_style(os.path.join(RESULTS, "rl_vs_bc"), 0)
    versus_bc = replay_style(os.path.join(RESULTS, "rl_vs_bc"), 1)
    imitation_curve()
    rl_curves(results["ganon_bc"]["win"], results["cpu_bc"]["lead"])
    style_figure(human, {**bc, **{k: versus_bc[k] for k in ("deaths", "self_destructs")}},
                 {**rl, **{k: versus_rl[k] for k in ("deaths", "self_destructs")}})
    screenshots()
    numbers = write_numbers(results, human, bc, rl, versus_bc, versus_rl, rl_hours)
    print(f"snapshot {snapshot} ({rl_hours:.1f} h of RL); replays read: imitation vs CPU {bc['games']}, "
          f"RL vs CPU {rl['games']}, RL vs imitation {versus_rl['games']}")
    for name, value in numbers.items():
        print(f"  {name} = {value}")
    print("figures and fig/results.tex written; now build the PDF with poster/build.ps1")


if __name__ == "__main__":
    main()
