"""Live check: a game converted live (libmelee) and from its saved replay (peppi-py)
gives the same raw frames, field by field.

Plays one full game of random controllers against a level 9 CPU with replay saving on,
then parses the replay Dolphin wrote and compares every frame both sides saw.

Usage: python tests/live/check_replay_vs_live.py
"""

import glob
import os
import sys
import tempfile
import time

import numpy as np
import peppi_py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smash_rl.codec import ACTION_SIZES  # noqa: E402
from smash_rl.env import SmashEnv  # noqa: E402
from smash_rl.state import PLAYER_DTYPE, from_peppi  # noqa: E402


def main():
    out_dir = tempfile.mkdtemp(prefix="smash_replay_check_")
    env = SmashEnv(opponent="FOX", cpu_level=9, delay=1, port=51641, save_replays=out_dir)
    rng = np.random.default_rng(0)
    live = {}
    try:
        obs, _ = env.reset()
        live[obs["frame"]] = obs["players"]
        for _ in range(20000):  # a random agent loses 4 stocks well before this
            action = np.array([rng.integers(0, size) for size in ACTION_SIZES])
            obs, _, done, _, _ = env.step(action)
            if done:
                break
            live[obs["frame"]] = obs["players"]
        else:
            raise SystemExit("the game did not end; run it again")
        # Dolphin waits for our inputs, so keep playing through the end of the game (the
        # next reset walks the menus); Slippi finishes the replay only then.
        env.reset()
        time.sleep(1)
    finally:
        env.close()
    replays = sorted(glob.glob(os.path.join(out_dir, "**", "*.slp"), recursive=True), key=os.path.getmtime)
    if not replays:
        raise SystemExit(f"no replay found in {out_dir}")
    players, stage, frame_ids = from_peppi(peppi_py.read_slippi(replays[0]))
    by_frame = {int(f): i for i, f in enumerate(frame_ids)}
    common = sorted(f for f in live if f in by_frame)
    print(f"{len(live)} live frames, {len(frame_ids)} replay frames, {len(common)} in both; stage index {stage}")
    bad = 0
    for name in PLAYER_DTYPE.names:
        a = np.stack([live[f][name] for f in common]).astype(np.float64)
        b = np.stack([players[by_frame[f]][name] for f in common]).astype(np.float64)
        wrong = ~np.isclose(a, b, atol=1e-4)
        if wrong.any():
            bad += 1
            i = np.argwhere(wrong)[0]
            print(f"  {name}: {wrong.sum()} mismatches, first at frame {common[i[0]]} port {i[1] + 1}: "
                  f"live {a[tuple(i)]} replay {b[tuple(i)]}")
    print("all fields match" if not bad and len(common) > 1000 else f"{bad} fields differ")
    sys.exit(0 if not bad and len(common) > 1000 else 1)


if __name__ == "__main__":
    main()
