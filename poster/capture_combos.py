"""Screenshots of the bot's combos for the poster, captured from the Dolphin window (Windows).

    python poster/capture_combos.py models/fox_rl_3h.pt --combos 4

Plays the bot against the level 9 CPU in a large window, one frame at a time (Dolphin waits
for every input, so each picture belongs to a known game frame), keeps the last seconds of
frames and saves each combo of --min-hits hits with --min-moves different moves into
poster/frames/combo<N>/: the frames as JPEG and hits.json (frame, move and damage of each hit).
"""

import argparse
import collections
import ctypes
import ctypes.wintypes as wt
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from smash_rl.agent import Agent  # noqa: E402
from smash_rl.env import SmashEnv  # noqa: E402
from smash_rl.model import load_policy  # noqa: E402

MOVE_NAMES = {44: "jab", 50: "dash attack", 51: "forward-tilt", 52: "forward-tilt", 53: "forward-tilt",
              54: "forward-tilt", 55: "forward-tilt", 56: "up-tilt", 57: "down-tilt", 58: "forward-smash",
              59: "forward-smash", 60: "forward-smash", 61: "forward-smash", 62: "forward-smash", 63: "up-smash",
              64: "down-smash", 65: "neutral-air", 66: "forward-air", 67: "back-air", 68: "up-air", 69: "down-air",
              219: "forward-throw", 220: "back-throw", 221: "up-throw", 222: "down-throw",
              360: "shine", 361: "shine", 362: "shine", 363: "shine", 364: "shine", 365: "shine",
              349: "side-B", 350: "side-B", 353: "up-B", 354: "up-B", 341: "laser", 344: "laser"}
COMBO_GAP = 45  # frames: the next hit within this keeps the combo going
MARGIN = 40  # frames kept before the first and after the last hit
user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


def game_window(pid: int):
    """(hwnd, width, height) of the Dolphin render window (titled "Dolphin") of process pid."""
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def visit(hwnd, _):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        title = ctypes.create_unicode_buffer(64)
        user32.GetWindowTextW(hwnd, title, 64)
        if owner.value == pid and user32.IsWindowVisible(hwnd) and title.value == "Dolphin":
            rect = wt.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            found.append((hwnd, rect.right, rect.bottom))
        return True

    user32.EnumWindows(visit, 0)
    if not found:
        raise RuntimeError("no Dolphin game window found")
    return found[0]


def grab(hwnd, width: int, height: int) -> np.ndarray:
    """The window's picture (height, width, 3); works when other windows cover it."""
    hdc = user32.GetDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(hwnd, mem, 3)  # client area, full GPU-rendered content
    info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(mem, bmp, 0, height, buf, ctypes.byref(info), 0)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[..., [2, 1, 0]]


def jpeg(image: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(out, "JPEG", quality=93)
    return out.getvalue()


def parse_args():
    p = argparse.ArgumentParser(description="Capture screenshots of the bot's combos for the poster")
    p.add_argument("checkpoint", help="the bot (.pt)")
    p.add_argument("--combos", type=int, default=4, help="stop after saving this many combos")
    p.add_argument("--max-games", type=int, default=4)
    p.add_argument("--min-hits", type=int, default=4)
    p.add_argument("--min-moves", type=int, default=3, help="different moves in a combo")
    p.add_argument("--cpu-level", type=int, default=9)
    p.add_argument("--opponent", default="FOX", help="CPU character")
    p.add_argument("--out", default=os.path.join("poster", "frames"))
    p.add_argument("--port", type=int, default=51600)
    return p.parse_args()


def main():
    args = parse_args()
    user32.SetProcessDPIAware()  # real pixel sizes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(load_policy(args.checkpoint, device), 1, device)
    env = SmashEnv(opponent=args.opponent, cpu_level=args.cpu_level, port=args.port, window_size=(1280, 960),
                   gfx_settings={"Settings": {"EFBScale": 4, "ShowFPS": False}})
    saved = len(glob_combos(args.out))
    try:
        for game in range(args.max_games):
            obs, _ = env.reset()
            agent.reset(0)
            hwnd, width, height = game_window(env.dolphin.console._process.pid)
            frames = collections.deque(maxlen=400)  # (game frame, jpeg)
            actions = collections.deque(maxlen=400)  # (game frame, our action state)
            hits, prev, done = [], obs["players"].copy(), False
            print(f"game {game + 1}: playing", flush=True)
            while not done:
                action = agent.act(obs["players"][None], np.array([obs["stage"]]))[0]
                obs, _, done, _, _ = env.step(action)
                f, players = obs["frame"], obs["players"]
                for _ in range(5):  # PrintWindow sometimes catches the window mid-redraw (black)
                    picture = grab(hwnd, width, height)
                    if picture.mean() > 8:
                        break
                    time.sleep(0.004)
                frames.append((f, jpeg(picture)))
                actions.append((f, int(players[0]["action"])))
                if players[1]["percent"] > prev[1]["percent"] and players[1]["stocks"] == prev[1]["stocks"]:
                    recent = [a for fr, a in actions if fr > f - 7]
                    move = next((MOVE_NAMES[a] for a in reversed(recent) if a in MOVE_NAMES), "other")
                    hits.append({"frame": int(f), "move": move, "percent": float(players[1]["percent"])})
                prev = players.copy()
                if hits and (f - hits[-1]["frame"] > COMBO_GAP or done):
                    moves = {h["move"] for h in hits} - {"other"}
                    if len(hits) >= args.min_hits and len(moves) >= args.min_moves:
                        saved += 1
                        folder = os.path.join(args.out, f"combo{saved}")
                        os.makedirs(folder, exist_ok=True)
                        first, last = hits[0]["frame"] - MARGIN, hits[-1]["frame"] + MARGIN
                        for fr, data in frames:
                            if first <= fr <= last:
                                with open(os.path.join(folder, f"frame{fr:06d}.jpg"), "wb") as out:
                                    out.write(data)
                        with open(os.path.join(folder, "hits.json"), "w") as out:
                            json.dump(hits, out, indent=1)
                        print(f"  combo {saved}: {len(hits)} hits, {' > '.join(h['move'] for h in hits)}", flush=True)
                        if saved >= args.combos:
                            return
                    hits = []
    finally:
        env.close()


def glob_combos(folder: str) -> list:
    return [d for d in (os.listdir(folder) if os.path.isdir(folder) else []) if d.startswith("combo")]


if __name__ == "__main__":  # libmelee's helper process re-imports this file
    main()
