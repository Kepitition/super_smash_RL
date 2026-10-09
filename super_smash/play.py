"""Play against a trained agent with an Xbox controller or the keyboard (Linux).

Usage:
    python play.py runs/<run>/final_model.zip                      # Xbox controller plugged in by USB
    python play.py runs/<run>/final_model.zip --input keyboard
    python play.py runs/<run>/final_model.zip --smash-hold-ms 200  # hold A longer before it smashes

The agent is port 1 and picks Fox; you are port 2. On the character select screen
pick your character and press Start; the agent then picks Final Destination, the
only stage it knows. Games run at normal speed with sound and can be paused. Close
the Dolphin window or press Ctrl+C here to stop.

The Xbox controller is read here in Python (see smash_rl/xbox_pad.py) so that the
stick never jumps and A held with a direction smashes. The keyboard is mapped in
Dolphin: its mapping lives in dolphin/play_home and can be changed with Controllers >
Port 2 > Configure in the Dolphin window (kept until you pass --remap).
"""

import argparse
import configparser
import os

import melee
from stable_baselines3 import PPO

from smash_rl import xbox_pad
from smash_rl.env import DEFAULT_DOLPHIN, PROJECT_DIR, SmashEnv
from smash_rl.xbox_pad import XboxPad

PLAY_HOME = os.path.join(PROJECT_DIR, "dolphin", "play_home")
GCPAD_INI = os.path.join(PLAY_HOME, "Config", "GCPadNew.ini")

XBOX_LAYOUT = """\
  Xbox             GameCube
  left stick       control stick (pushing up never jumps)
  right stick      C-stick (smash attacks, aerials)
  A                A: tap for the normal attack, hold with a direction for a smash
  B                B (special)
  X / Y            X / Y (jump)
  RB               Z (grab)
  LT / RT          L / R (shield, air dodge)
  Menu             Start
  D-pad            D-pad"""

KEYBOARD_LAYOUT = """\
  Keyboard         GameCube
  W A S D          control stick
  hold Left Shift  half tilt (walk, tilt attacks)
  arrow keys       C-stick (smash attacks, aerials)
  J                A
  K                B
  L / I            X / Y (jump)
  O                Z (grab)
  Space            R (shield, air dodge)
  Enter            Start"""


class HumanEnv(SmashEnv):
    """SmashEnv whose port 2 is fed by an XboxPad; the pad sees every game state."""

    def __init__(self, pad: XboxPad, **kwargs):
        super().__init__(**kwargs)
        self.xbox = pad

    def _start_dolphin(self):
        super()._start_dolphin()
        self.xbox.attach(self.console)
        self.xbox.start()

    def _next_state(self):
        gs = super()._next_state()
        self.xbox.gamestate = gs
        return gs


def _read_gcpad_ini():
    cfg = configparser.ConfigParser()  # lower-cases keys like libmelee does; Dolphin ignores case
    cfg.read(GCPAD_INI)
    return cfg


def map_port2_to_pipe():
    """Map port 2 to libmelee's named pipe in PLAY_HOME; Dolphin has to see it when it starts."""
    cfg = _read_gcpad_ini()
    if cfg.remove_section("GCPad2"):
        with open(GCPAD_INI, "w") as f:
            cfg.write(f)
    console = melee.Console(path=DEFAULT_DOLPHIN, dolphin_home_path=PLAY_HOME, tmp_home_directory=False)
    console.setup_dolphin_controller(2)


def keyboard_controls() -> dict:
    """GCPadNew.ini mapping for the keyboard through Dolphin's XInput2 backend (X key names)."""
    return {
        "Device": "XInput2/0/Virtual core pointer",
        "Buttons/A": "`J`",
        "Buttons/B": "`K`",
        "Buttons/X": "`L`",
        "Buttons/Y": "`I`",
        "Buttons/Z": "`O`",
        "Buttons/Start": "`Return`",
        "Main Stick/Up": "`W`",
        "Main Stick/Down": "`S`",
        "Main Stick/Left": "`A`",
        "Main Stick/Right": "`D`",
        "Main Stick/Modifier": "`Shift_L`",
        "Main Stick/Modifier/Range": "50.0",
        "C-Stick/Up": "`Up`",
        "C-Stick/Down": "`Down`",
        "C-Stick/Left": "`Left`",
        "C-Stick/Right": "`Right`",
        "Triggers/R": "`space`",
        "Triggers/R-Analog": "`space`",
    }


def write_keyboard_controls(remap: bool) -> bool:
    """Map port 2 to the keyboard. Returns False if an existing keyboard mapping was kept."""
    controls = keyboard_controls()
    os.makedirs(os.path.dirname(GCPAD_INI), exist_ok=True)
    cfg = _read_gcpad_ini()
    if not remap and cfg.has_section("GCPad2") and cfg.get("GCPad2", "Device", fallback="") == controls["Device"]:
        return False
    cfg.remove_section("GCPad2")
    cfg.add_section("GCPad2")
    for key, value in controls.items():
        cfg.set("GCPad2", key, value)
    with open(GCPAD_INI, "w") as f:
        cfg.write(f)
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Play Melee against a trained PPO agent")
    p.add_argument("model", help="path to a saved .zip model")
    p.add_argument("--input", choices=("xbox", "keyboard"), default="xbox", help="what you play with")
    p.add_argument("--remap", action="store_true", help="keyboard: overwrite your mapping with the default one")
    p.add_argument("--smash-hold-ms", type=int, default=150,
                   help="Xbox: how long A must be held with a direction to smash instead of the normal attack")
    p.add_argument("--character", default="FOX", help="the agent's character, must match training")
    p.add_argument("--frame-skip", type=int, default=3, help="must match training")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely one")
    p.add_argument("--port", type=int, default=51520)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(PLAY_HOME, exist_ok=True)
    env_kwargs = dict(character=args.character, opponent="HUMAN", frame_skip=args.frame_skip, port=args.port,
                      emulation_speed=1.0, dolphin_home=PLAY_HOME)
    pad = None
    if args.input == "xbox":
        name, device = xbox_pad.find_pad()
        pad = XboxPad(device, smash_hold=args.smash_hold_ms / 1000)
        map_port2_to_pipe()
        print(f"port 2: {name} ({device})\n{XBOX_LAYOUT}")
        env = HumanEnv(pad, **env_kwargs)
    else:
        written = write_keyboard_controls(args.remap)
        print(f"port 2: keyboard ({'default mapping' if written else 'your saved mapping'})")
        if written:
            print(KEYBOARD_LAYOUT)
        env = SmashEnv(**env_kwargs)

    model = PPO.load(args.model, device="cpu")
    if model.observation_space.shape != env.observation_space.shape:
        raise SystemExit(f"{args.model} expects observations of shape {model.observation_space.shape}, "
                         f"but the game gives {env.observation_space.shape}: it was trained with an older env.py")

    results = []
    try:
        while True:
            print("\npick your character on port 2 and press Start...")
            obs, info = env.reset()
            print(f"game {len(results) + 1}: you ({info['opponent']}) vs the agent ({args.character})")
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, _, done, _, info = env.step(action)
            results.append(info)
            print(f"  {'the agent' if info['won'] else 'you'} won | your stock difference {-info['stock_diff']:+d} | "
                  f"you dealt {info['damage_taken']:.0f}%, took {info['damage_dealt']:.0f}%")
    except KeyboardInterrupt:
        pass
    except Exception as e:  # closing the Dolphin window ends the connection
        print(f"\nDolphin stopped ({type(e).__name__}: {e})")
    finally:
        env.close()
        if pad is not None:
            pad.close()
        if results:
            yours = sum(1 for r in results if not r["won"])
            print(f"\nyou won {yours} of {len(results)} games")


if __name__ == "__main__":
    main()
