"""Ranked replays (erickfm/melee-ranked-replays): filtering and conversion to raw frames."""

import hashlib
import json
import os

import numpy as np
import peppi_py

from smash_rl.state import STAGE_FROM_SLIPPI, MissingData, from_peppi

REPO = "erickfm/melee-ranked-replays"
MIN_PLAY_FRAMES = 1800  # 30 s after the countdown
TIMEOUT_FRAMES = 8 * 60 * 60 - 60  # ranked games last at most 8 minutes
NO_CONTEST = 7  # peppi EndMethod of a quit-out (LRAS)


class Skip(Exception):
    """A replay we don't train on; the message is the reason."""


def check_game(game, character: int) -> list:
    """Slots (0 or 1) where `character` plays in a game we keep; raises Skip otherwise."""
    if len(game.start.players) != 2 or len(game.frames.ports) != 2:
        raise Skip("not 2 players")
    if any(int(p.type) != 0 for p in game.start.players):
        raise Skip("not human")
    if int(game.start.stage) not in STAGE_FROM_SLIPPI:
        raise Skip("stage")
    # The ranked dumps carry no end record, so a finished game is one where a player has
    # no stocks left on the last frame or the 8-minute timer ran out.
    frame_ids = np.unique(np.asarray(game.frames.id))  # rollbacks repeat frames
    last_stocks = [int(np.asarray(port.leader.post.stocks)[-1]) for port in game.frames.ports]
    ran_out_of_time = int(frame_ids.max()) >= TIMEOUT_FRAMES
    if ((game.end is not None and int(game.end.method) == NO_CONTEST)
            or (min(last_stocks) > 0 and not ran_out_of_time)):
        raise Skip("quit or disconnect")
    if int((frame_ids >= 0).sum()) < MIN_PLAY_FRAMES:
        raise Skip("short")
    slots = [i for i, port in enumerate(game.frames.ports)
             if int(np.asarray(port.leader.post.character)[0]) == character]
    if not slots:
        raise Skip("character absent")
    return slots


def game_split(game_id: str) -> str:
    return "val" if int(hashlib.sha1(game_id.encode()).hexdigest(), 16) % 100 < 2 else "train"


def convert_safely(slp_bytes: bytes, game_id: str, out_path: str, character: int) -> dict:
    """Parse one replay and write its raw frames to out_path (.npz). Runs in a worker process.

    Returns the index entry (without archive/rank/path), or {"skip": reason}.
    """
    tmp = out_path + ".slp.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(slp_bytes)
        game = peppi_py.read_slippi(tmp)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # also pyo3's PanicException, which a malformed replay can raise
        return {"skip": "parse error"}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    try:
        slots = check_game(game, character)
        players, stage, frame_ids = from_peppi(game)
    except Skip as e:
        return {"skip": str(e)}
    except MissingData:
        return {"skip": "missing data"}
    except Exception as e:  # an odd replay must not stop the whole run
        return {"skip": f"bad replay ({type(e).__name__})"}
    np.savez_compressed(out_path, players=players, stage=np.int8(stage), first_frame=np.int32(frame_ids[0]))
    return {"game_id": game_id, "stage": int(stage), "characters": [int(c) for c in players[0]["character"]],
            "slots": slots, "frames": int(len(players)), "split": game_split(game_id)}


def rewrite_index(path: str, entries: list):
    """Replace the index atomically, so an interruption never leaves it half written."""
    with open(path + ".tmp", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)
    os.replace(path + ".tmp", path)


def read_index(path: str) -> list:
    """Index entries; a half-written last line (an interrupted run) is dropped."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return entries
