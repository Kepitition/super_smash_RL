"""Live check: every controller class sent through the pipe reads back as the same class.

Plays a game against a level 1 CPU with delay 1, so the decision made after frame t is
the input of frame t+1, and snaps the controller the game recorded for that frame.
Stick classes outside the stick's circle (the game clamps them), the ±0.125 classes
(inside the deadzone, never produced by replays) and Z with a light trigger (the game
reports Z as a light trigger press anyway, so it reads back as plain Z) are not expected
to round-trip. A few neutral frames first skip the game's first frame, which libmelee
answers with its own neutral input.

Usage: python tests/live/check_controller_roundtrip.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smash_rl.codec import STICK_SEND, controller_classes  # noqa: E402
from smash_rl.env import SmashEnv  # noqa: E402

FIELDS = ["main_x", "main_y", "c_x", "c_y", "trigger", "buttons"]


def cases():
    for buttons in range(48):
        if not (buttons & 8 and buttons >> 4 == 1):  # Z + light trigger
            yield (buttons, 8, 8, 0)
    for x in range(17):
        for y in range(17):
            if x not in (7, 9) and y not in (7, 9) and STICK_SEND[x] ** 2 + STICK_SEND[y] ** 2 <= 1 + 1e-9:
                yield (0, x, y, 0)
    for c in range(5):
        yield (0, 8, 8, c)


def main():
    env = SmashEnv(opponent="FOX", cpu_level=1, delay=1, port=51640)
    failures, total = [], 0
    try:
        env.reset()
        for _ in range(5):
            env.step(np.array([0, 8, 8, 0]))
        for case in cases():
            total += 1
            obs, _, done, _, _ = env.step(np.array(case))
            if done:
                raise SystemExit("the game ended during the check; run it again")
            got = tuple(int(v) for v in controller_classes(obs["players"][:1])[0])
            if got != case:
                failures.append((case, got, obs["players"][0][FIELDS]))
    finally:
        env.close()
    print(f"{total - len(failures)}/{total} controller classes read back unchanged")
    for case, got, raw in failures[:40]:
        print(f"  sent {case} -> read {got}  raw {raw}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
