"""Play against a trained bot with an Xbox controller.

Usage:
    python play.py runs/bc_fox/bc_fox_10h.pt
    python play.py runs/bc_fox/bc_fox_10h.pt --stage BATTLEFIELD
    python play.py runs/bc_fox/bc_fox_10h.pt --remap      # restore the default Xbox mapping

The bot is port 1 and plays Fox; you are port 2. On the character select screen pick
your character and press Start; the bot then picks the stage. Games run at normal speed
with sound and can be paused. Close the Dolphin window or press Ctrl+C here to stop.

Your controls live in dolphin/play_home (kept between runs): change them in the Dolphin
window under Controllers > Port 2 > Configure; --remap writes the default mapping again.
"""

import argparse
import configparser
import os

import numpy as np
import torch

from smash_rl.agent import Agent
from smash_rl.dolphin import PROJECT_DIR
from smash_rl.env import SmashEnv
from smash_rl.model import load_policy

PLAY_HOME = os.path.join(PROJECT_DIR, "dolphin", "play_home")
GCPAD_INI = os.path.join(PLAY_HOME, "Config", "GCPadNew.ini")
XBOX_DEVICE = "XInput/0/Gamepad"
XBOX_LAYOUT = """\
  Xbox             GameCube
  left stick       control stick
  right stick      C-stick (smash attacks, aerials)
  A / B            A / B
  X / Y            X / Y (jump)
  RB               Z (grab)
  LT / RT          L / R (shield, air dodge, L-cancel)
  Menu             Start
  D-pad            D-pad"""


def xbox_mapping() -> dict:
    """GCPadNew.ini keys for an Xbox pad on Dolphin's XInput backend (lower-case, like libmelee)."""
    return {
        "device": XBOX_DEVICE,
        "buttons/a": "`Button A`", "buttons/b": "`Button B`",
        "buttons/x": "`Button X`", "buttons/y": "`Button Y`",
        "buttons/z": "`Shoulder R`", "buttons/start": "`Start`",
        "main stick/up": "`Left Y+`", "main stick/down": "`Left Y-`",
        "main stick/left": "`Left X-`", "main stick/right": "`Left X+`",
        "c-stick/up": "`Right Y+`", "c-stick/down": "`Right Y-`",
        "c-stick/left": "`Right X-`", "c-stick/right": "`Right X+`",
        "triggers/l": "`Trigger L`", "triggers/r": "`Trigger R`",
        "triggers/l-analog": "`Trigger L`", "triggers/r-analog": "`Trigger R`",
        "d-pad/up": "`Pad N`", "d-pad/down": "`Pad S`", "d-pad/left": "`Pad W`", "d-pad/right": "`Pad E`",
    }


def write_xbox_mapping(ini_path: str, remap: bool) -> bool:
    """Map port 2 to the Xbox pad. Returns False if an existing Xbox mapping was kept."""
    # Lower-case keys (ConfigParser's default) like libmelee writes them: Dolphin ignores
    # case, but mixing both would leave duplicate keys behind in this kept folder.
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    if not remap and cfg.has_section("GCPad2") and cfg.get("GCPad2", "device", fallback="") == XBOX_DEVICE:
        return False
    cfg.remove_section("GCPad2")
    cfg.add_section("GCPad2")
    for key, value in xbox_mapping().items():
        cfg.set("GCPad2", key, value)
    os.makedirs(os.path.dirname(ini_path), exist_ok=True)
    with open(ini_path, "w") as f:
        cfg.write(f)
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Play Melee against a trained bot with an Xbox controller")
    p.add_argument("checkpoint", help="a .pt file from train_bc.py")
    p.add_argument("--stage", default="FINAL_DESTINATION", help="stage the bot picks, e.g. BATTLEFIELD")
    p.add_argument("--remap", action="store_true", help="overwrite your port 2 mapping with the default Xbox one")
    p.add_argument("--delay", type=int, default=18, help="the bot's reaction delay in frames, must match training")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (1 = as trained)")
    p.add_argument("--port", type=int, default=51520, help="Slippi port of this Dolphin")
    return p.parse_args()


def main():
    args = parse_args()
    written = write_xbox_mapping(GCPAD_INI, args.remap)
    print(f"port 2: Xbox controller ({'default mapping' if written else 'your saved mapping'})")
    if written:
        print(XBOX_LAYOUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(load_policy(args.checkpoint, device), 1, device, args.temperature)
    env = SmashEnv(opponent="HUMAN", stages=(args.stage,), delay=args.delay, port=args.port,
                   emulation_speed=1.0, dolphin_home=PLAY_HOME)
    results = []
    try:
        while True:
            print("\npick your character on port 2 and press Start...")
            obs, info = env.reset()
            agent.reset(0)
            print(f"game {len(results) + 1}: you ({info['opponent']}) vs the bot (FOX)")
            done = False
            while not done:
                action = agent.act(obs["players"][None], np.array([obs["stage"]]))[0]
                obs, _, done, _, info = env.step(action)
            results.append(info)
            print(f"  {'the bot' if info['won'] else 'you'} won | your stock difference {-info['stock_diff']:+d} | "
                  f"you dealt {info['damage_taken']:.0f}%, took {info['damage_dealt']:.0f}%")
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:  # closing the Dolphin window ends the session
        print(f"\nstopped: {e}")
    finally:
        env.close()
        if results:
            print(f"\nyou won {sum(1 for r in results if not r['won'])} of {len(results)} games")


if __name__ == "__main__":
    main()
