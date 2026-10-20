"""Controller classes and model inputs, both computed from raw frames (smash_rl.state).

The agent picks a controller as four classes, in this order (the model conditions
each on the earlier ones): buttons (A, B, jump, Z, trigger level), main stick x,
main stick y, C-stick direction. Replays are snapped to these classes for the
training labels; live, a class tuple is decoded and sent to Dolphin.
"""

import melee
import numpy as np

from smash_rl.state import BUTTON_BITS, NUM_STAGES

_B = melee.Button
NUM_BUTTON_CLASSES = 48  # A x B x jump x Z (2 each) x trigger (none, light, full)
STICK_STEP = 0.125
NUM_STICK_CLASSES = 17  # -1 .. 1 in steps of STICK_STEP
NUM_CSTICK_CLASSES = 5  # neutral, up, down, left, right
ACTION_SIZES = (NUM_BUTTON_CLASSES, NUM_STICK_CLASSES, NUM_STICK_CLASSES, NUM_CSTICK_CLASSES)
NEUTRAL = np.array([0, 8, 8, 0], dtype=np.int64)

# The game counts a trigger as lightly pressed from 43/140 = 0.307 of its travel.
LIGHT_TRIGGER = 0.3
CSTICK_THRESHOLD = 0.3
# What each stick class is sent as. ±0.25 lies inside Melee's deadzone (0.2875), so it
# is sent as ±0.3, the smallest tilt the game registers, which snaps back to ±0.25.
STICK_SEND = np.round(np.arange(NUM_STICK_CLASSES) * STICK_STEP - 1.0, 4)
STICK_SEND[6], STICK_SEND[10] = -0.3, 0.3
CSTICK_SEND = ((0.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0))
TRIGGER_SEND = (0.0, 0.35, 1.0)  # none, light, full (full also presses digital L)


def button_class(a, b, jump, z, trigger):
    i = lambda v: np.asarray(v, dtype=np.int64)
    return i(a) + 2 * i(b) + 4 * i(jump) + 8 * i(z) + 16 * i(trigger)


def button_parts(cls):
    """(a, b, jump, z, trigger) of button classes."""
    cls = np.asarray(cls, dtype=np.int64)
    return cls & 1, (cls >> 1) & 1, (cls >> 2) & 1, (cls >> 3) & 1, cls >> 4


def snap_axis(values) -> np.ndarray:
    v = (np.asarray(values, dtype=np.float64) + 1.0) / STICK_STEP
    return np.clip(np.floor(v + 0.5), 0, NUM_STICK_CLASSES - 1).astype(np.int64)


def controller_classes(frames) -> np.ndarray:
    """(..., 4) controller classes of raw frames (PLAYER_DTYPE records)."""
    bits = np.asarray(frames["buttons"]).astype(np.int64)

    def pressed(*buttons):
        return (bits & sum(BUTTON_BITS[b] for b in buttons)) != 0

    z = pressed(_B.BUTTON_Z)
    # Z also shows up as a light analog press in the processed trigger, so it only counts
    # without Z; a digital L or R is a full press.
    trigger = np.where(pressed(_B.BUTTON_L, _B.BUTTON_R), 2,
                       np.where((np.asarray(frames["trigger"]) >= LIGHT_TRIGGER) & ~z, 1, 0))
    buttons = button_class(pressed(_B.BUTTON_A), pressed(_B.BUTTON_B),
                           pressed(_B.BUTTON_X, _B.BUTTON_Y), z, trigger)
    cx, cy = np.asarray(frames["c_x"]), np.asarray(frames["c_y"])
    cstick = np.where(np.maximum(np.abs(cx), np.abs(cy)) < CSTICK_THRESHOLD, 0,
                      np.where(np.abs(cy) >= np.abs(cx), np.where(cy > 0, 1, 2), np.where(cx < 0, 3, 4)))
    return np.stack([buttons, snap_axis(frames["main_x"]), snap_axis(frames["main_y"]), cstick],
                    axis=-1).astype(np.int64)


def apply_controller(pad, action) -> None:
    """Set a libmelee controller to a class tuple; it is sent on the next flush."""
    a, b, jump, z, trigger = (int(v) for v in button_parts(int(action[0])))
    for button, on in ((_B.BUTTON_A, a), (_B.BUTTON_B, b), (_B.BUTTON_X, jump),
                       (_B.BUTTON_Z, z), (_B.BUTTON_L, trigger == 2)):
        if on:
            pad.press_button(button)
        else:
            pad.release_button(button)
    pad.press_shoulder(_B.BUTTON_L, TRIGGER_SEND[trigger])
    pad.tilt_analog_unit(_B.BUTTON_MAIN, STICK_SEND[int(action[1])], STICK_SEND[int(action[2])])
    pad.tilt_analog_unit(_B.BUTTON_C, *CSTICK_SEND[int(action[3])])


NUM_ACTIONS = 400  # action-state ids; a few character-specific ones above share the last slot
NUM_CHARACTERS = 33
NUM_JUMPS = 8
NUMERIC_PER_PLAYER = 16


def encode_state(players, stage) -> dict:
    """Model inputs from raw frames (..., 2) [agent, opponent] and stage indices (...)."""
    p = players
    numeric = np.stack([
        p["x"] * 0.05, p["y"] * 0.05, p["percent"] * 0.01, p["stocks"] / 4.0,
        np.where(p["facing"], 1.0, -1.0), p["on_ground"], p["invulnerable"],
        np.clip(p["hitlag"] / 30.0, 0, 4), np.clip(p["hitstun"] / 30.0, 0, 4),
        np.clip(p["action_frame"] / 60.0, 0, 2), p["shield"] * 0.01,
        np.clip(p["speed_air_x"] * 0.5, -10, 10), np.clip(p["speed_y"] * 0.5, -10, 10),
        np.clip(p["speed_attack_x"] * 0.5, -10, 10), np.clip(p["speed_attack_y"] * 0.5, -10, 10),
        np.clip(p["speed_ground_x"] * 0.5, -10, 10),
    ], axis=-1)
    # One NaN would poison the LSTM state of its batch row for the rest of a training run.
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    return {
        "numeric": numeric.reshape(numeric.shape[:-2] + (2 * NUMERIC_PER_PLAYER,)),
        "action": np.minimum(p["action"], NUM_ACTIONS - 1).astype(np.int64),
        "character": np.minimum(p["character"], NUM_CHARACTERS - 1).astype(np.int64),
        "jumps": np.minimum(p["jumps_left"], NUM_JUMPS - 1).astype(np.int64),
        "stage": np.broadcast_to(np.minimum(np.asarray(stage, dtype=np.int64), NUM_STAGES - 1),
                                 players.shape[:-1]).copy(),
    }
