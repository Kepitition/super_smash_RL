"""Raw per-frame game state: the one format shared by replays and live games.

`from_peppi` reads a parsed Slippi replay (peppi-py) and `from_libmelee` a live
libmelee GameState. Both fill the same fields the way libmelee does (truncated
hitlag and hitstun, action frames shifted to 1-indexed for some actions), so
anything computed from either source matches. smash_rl.codec derives the model
inputs and controller classes from these records.
"""

import csv
import os

import melee
import numpy as np

PLAYER_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("percent", "f4"), ("stocks", "u1"),
    ("facing", "?"), ("on_ground", "?"), ("invulnerable", "?"),
    ("hitlag", "f4"), ("hitstun", "f4"), ("jumps_left", "u1"), ("shield", "f4"),
    ("action", "u2"), ("action_frame", "f4"), ("character", "u1"),
    ("speed_air_x", "f4"), ("speed_y", "f4"), ("speed_attack_x", "f4"),
    ("speed_attack_y", "f4"), ("speed_ground_x", "f4"),
    # Controller as the game sees it: processed sticks (-1..1) and analog trigger
    # (0..1, L and R combined), and the physical buttons as BUTTON_BITS.
    ("main_x", "f4"), ("main_y", "f4"), ("c_x", "f4"), ("c_y", "f4"),
    ("trigger", "f4"), ("buttons", "u2"),
])

LEGAL_STAGES = (
    melee.Stage.FINAL_DESTINATION, melee.Stage.BATTLEFIELD, melee.Stage.DREAMLAND,
    melee.Stage.YOSHIS_STORY, melee.Stage.FOUNTAIN_OF_DREAMS, melee.Stage.POKEMON_STADIUM,
)
OTHER_STAGE = len(LEGAL_STAGES)
NUM_STAGES = OTHER_STAGE + 1
# Slippi's game-start stage ids of LEGAL_STAGES, in the same order.
STAGE_FROM_SLIPPI = {sid: i for i, sid in enumerate((0x20, 0x1F, 0x1C, 0x08, 0x02, 0x03))}
_STAGE_FROM_LIBMELEE = {stage: i for i, stage in enumerate(LEGAL_STAGES)}

_B = melee.Button
BUTTON_BITS = {
    _B.BUTTON_A: 0x0100, _B.BUTTON_B: 0x0200, _B.BUTTON_X: 0x0400, _B.BUTTON_Y: 0x0800,
    _B.BUTTON_START: 0x1000, _B.BUTTON_Z: 0x0010, _B.BUTTON_R: 0x0020, _B.BUTTON_L: 0x0040,
    _B.BUTTON_D_LEFT: 0x0001, _B.BUTTON_D_RIGHT: 0x0002, _B.BUTTON_D_DOWN: 0x0004, _B.BUTTON_D_UP: 0x0008,
}
FOX = melee.Character.FOX.value


class MissingData(ValueError):
    """The replay lacks a field this format needs (Slippi versions before 3.5)."""


def _zero_indexed_actions() -> np.ndarray:
    """[character, action] -> whether libmelee adds 1 to the action frame (its actiondata.csv)."""
    table = np.zeros((256, 1024), dtype=bool)
    with open(os.path.join(os.path.dirname(melee.__file__), "actiondata.csv")) as f:
        for row in csv.DictReader(f):
            if row["zeroindex"] == "True":
                table[int(row["character"]), int(row["action"])] = True
    return table


ZERO_INDEXED = _zero_indexed_actions()


def stage_index(stage: melee.Stage) -> int:
    return _STAGE_FROM_LIBMELEE.get(stage, OTHER_STAGE)


def from_libmelee(gamestate: melee.GameState, ports=(1, 2)) -> np.ndarray:
    """Raw frames (len(ports),) of a live game state, in the order of `ports`."""
    out = np.zeros(len(ports), dtype=PLAYER_DTYPE)
    for i, port in enumerate(ports):
        p = gamestate.players[port]
        c = p.controller_state
        out[i] = (
            p.position.x, p.position.y, p.percent, p.stock,
            bool(p.facing), bool(p.on_ground), bool(p.invulnerable),
            p.hitlag_left, p.hitstun_frames_left, p.jumps_left, p.shield_strength,
            getattr(p.action, "value", p.action), p.action_frame, getattr(p.character, "value", p.character),
            p.speed_air_x_self, p.speed_y_self, p.speed_x_attack, p.speed_y_attack, p.speed_ground_x_self,
            (c.main_stick[0] - 0.5) * 2, (c.main_stick[1] - 0.5) * 2,
            (c.c_stick[0] - 0.5) * 2, (c.c_stick[1] - 0.5) * 2,
            c.l_shoulder, sum(bit for button, bit in BUTTON_BITS.items() if c.button.get(button)),
        )
    return out


def _column(values, dtype) -> np.ndarray:
    if values is None:
        raise MissingData("replay is missing a frame field")
    return np.asarray(values).astype(dtype, copy=False)


def _truncated(values) -> np.ndarray:
    """int() of each value like libmelee, with NaN and infinities as 0."""
    v = np.trunc(_column(values, np.float64))
    v[~np.isfinite(v)] = 0
    return v.astype(np.float32)


def from_peppi(game) -> tuple:
    """(raw frames (T, n_ports), stage index, frame ids (T,)) of a peppi-py Game.

    Online replays contain rollbacks: frames re-sent after a misprediction. Only the
    last copy of each frame (the one that stood) is kept, so rows are consecutive frames.
    """
    raw_ids = _column(game.frames.id, np.int32)
    _, last_in_reversed = np.unique(raw_ids[::-1], return_index=True)
    keep = len(raw_ids) - 1 - last_in_reversed  # in increasing frame order
    frame_ids = raw_ids[keep]
    if np.any(np.diff(frame_ids) != 1):
        raise MissingData("replay skips frames")
    players = np.zeros((len(raw_ids), len(game.frames.ports)), dtype=PLAYER_DTYPE)
    for i, port in enumerate(game.frames.ports):
        post, pre, col = port.leader.post, port.leader.pre, players[:, i]
        if post.velocities is None:
            raise MissingData("replay has no speeds (Slippi < 3.5)")
        character = _column(post.character, np.uint8)
        action = _column(post.state, np.uint16)
        col["x"] = _column(post.position.x, np.float32)
        col["y"] = _column(post.position.y, np.float32)
        col["percent"] = _column(post.percent, np.float32)
        col["stocks"] = _column(post.stocks, np.uint8)
        col["facing"] = _column(post.direction, np.float32) > 0
        col["on_ground"] = ~_column(post.airborne, bool)
        col["invulnerable"] = _column(post.hurtbox_state, np.uint8) != 0
        col["hitlag"] = _truncated(post.hitlag)
        col["hitstun"] = _truncated(post.misc_as)
        col["jumps_left"] = _column(post.jumps, np.uint8)
        col["shield"] = _column(post.shield, np.float32)
        col["action"] = action
        col["action_frame"] = _truncated(post.state_age) + ZERO_INDEXED[character, np.minimum(action, 1023)]
        col["character"] = character
        v = post.velocities
        col["speed_air_x"] = _column(v.self_x_air, np.float32)
        col["speed_y"] = _column(v.self_y, np.float32)
        col["speed_attack_x"] = _column(v.knockback_x, np.float32)
        col["speed_attack_y"] = _column(v.knockback_y, np.float32)
        col["speed_ground_x"] = _column(v.self_x_ground, np.float32)
        col["main_x"] = _column(pre.joystick.x, np.float32)
        col["main_y"] = _column(pre.joystick.y, np.float32)
        col["c_x"] = _column(pre.cstick.x, np.float32)
        col["c_y"] = _column(pre.cstick.y, np.float32)
        col["trigger"] = _column(pre.triggers, np.float32)
        col["buttons"] = _column(pre.buttons_physical, np.uint16)
    return players[keep], STAGE_FROM_SLIPPI.get(int(game.start.stage), OTHER_STAGE), frame_ids
