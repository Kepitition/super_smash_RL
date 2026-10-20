# Phase 1: Imitation-Learned Fox Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Fox bot learned from ~84k master-level human replays that plays every frame with a full controller and an 18-frame reaction delay, evaluated against the level 9 CPU.

**Architecture:** Replays (peppi-py) and live games (libmelee) are both converted to one raw per-frame record (`smash_rl/state.py`); one codec (`smash_rl/codec.py`) turns raw frames into model inputs and controller classes. The env steps one frame at a time and delays decisions through a queue. A recurrent policy (LSTM + feed-forward blocks, ~10M parameters, autoregressive controller head) is trained by behavior cloning on stateful 80-frame unrolls.

**Tech Stack:** Python 3.12 (Windows 11), PyTorch 2.11 + CUDA 12.8, libmelee 0.47.3, peppi-py 0.8.6, huggingface_hub, numpy 1.26, gymnasium 0.29, pytest, TensorBoard.

**Spec:** `docs/superpowers/specs/2026-10-14-phase1-imitation-fox-design.md`

## Global Constraints

- Run everything from the repo root with the venv: `.venv\Scripts\python.exe ...` (PowerShell) or `.venv/Scripts/python.exe ...` (Git Bash).
- No git commits: the user shares the repo themselves. Where the template would commit, the task ends with its verification instead.
- Character: Fox (`melee.Character.FOX`, internal id 1). Data: `erickfm/melee-ranked-replays`, folders `FOX/FOX_master-master_a{1..6}` and `FOX/FOX_master-diamond_a{1..6}`.
- Reaction delay: default 18, meaning a decision made after observing frame t is the controller input of frame t+18 (minimum 1).
- One decision per frame. Controller classes: buttons 48 (A × B × jump × Z × trigger {none, light, full}), main stick x 17, main stick y 17 (−1..1, step 0.125), C-stick 5 (neutral, up, down, left, right).
- Legal stages in model-index order: Final Destination 0, Battlefield 1, Dream Land 2, Yoshi's Story 3, Fountain of Dreams 4, Pokemon Stadium 5, other 6.
- Data directory default `D:\smash_data`; 2% of games (sha1 of the game id mod 100 < 2) are validation.
- Model defaults: width 512, 3 blocks, feed-forward ×2, head hidden 128. Imitation: batch 512, unroll 80, AdamW lr 1e-4, gradient clip 1.0, mixed precision.
- Code style: English, sparse "why" comments, every CLI option is an argparse flag with help text; README and AGENTS.md describe the flags and files.
- Live tests open Dolphin windows and use their own Slippi ports (51640+); check that no `Slippi Dolphin.exe` is left running afterwards.

## Review Focus

- **Replays from old Slippi versions** (missing speed or hitlag fields) must be skipped with a counted reason, never crash the data run. Tests: Task 1 (`from_peppi` raises `MissingData`), Task 5 (`convert_safely` reports `missing data`).
- **A corrupt or truncated `.slp` inside an archive** must be skipped as `parse error` while the archive keeps processing. Test: Task 5.
- **`prepare_data.py` killed mid-archive** (last index line half-written) must resume without crashing or duplicating games. Test: Task 5 (`read_index` drops a broken last line; known game ids are skipped).
- **Trajectories shorter than unroll + delay** must never be sampled, and a dataset with only short ones must raise instead of looping forever. Test: Task 8.
- **A game that ends with decisions still queued**: the next game must start from a neutral queue, not the previous game's last inputs. Test: Task 3 (`DelayQueue.reset`); `SmashEnv.reset` calls it.

---

### Task 1: Raw frame format (`smash_rl/state.py`)

**Files:**
- Create: `smash_rl/state.py`, `tests/test_state.py`, `pytest.ini`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `PLAYER_DTYPE` (numpy structured dtype, fields below), `LEGAL_STAGES`, `NUM_STAGES = 7`, `OTHER_STAGE = 6`, `STAGE_FROM_SLIPPI: dict[int, int]`, `BUTTON_BITS: dict[melee.Button, int]`, `FOX = 1`, `class MissingData(ValueError)`, `stage_index(stage: melee.Stage) -> int`, `from_libmelee(gamestate, ports=(1, 2)) -> np.ndarray` (shape `(len(ports),)`, `PLAYER_DTYPE`), `from_peppi(game) -> tuple[np.ndarray (T, n_ports) PLAYER_DTYPE, int stage index, np.ndarray (T,) int32 frame ids]`.

- [ ] **Step 1: Add dependencies and pytest config**

Append to `requirements.txt` (keep existing lines):

```text
# Replays and tests
peppi-py==0.8.6
pytest
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

Run: `.venv\Scripts\python.exe -m pip install peppi-py==0.8.6 pytest`
Expected: installs peppi-py 0.8.6 and pyarrow 21.x without errors.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_state.py`:

```python
from types import SimpleNamespace as NS

import melee
import numpy as np
import pytest

from smash_rl import state
from smash_rl.state import FOX, PLAYER_DTYPE, from_libmelee, from_peppi

A, X = state.BUTTON_BITS[melee.Button.BUTTON_A], state.BUTTON_BITS[melee.Button.BUTTON_X]


def fake_port(T, character=FOX, action=14, state_age=3.7, misc_as=10.9, velocities=True):
    f = lambda v, dtype=np.float32: np.full(T, v, dtype=dtype)
    post = NS(
        character=f(character, np.uint8), state=f(action, np.uint16),
        position=NS(x=f(-60.0), y=f(0.0)), direction=f(1.0), percent=f(12.5), shield=f(60.0),
        stocks=f(4, np.uint8), state_age=f(state_age), misc_as=f(misc_as), airborne=f(0, np.uint8),
        jumps=f(2, np.uint8), hurtbox_state=f(0, np.uint8), hitlag=f(2.5),
        velocities=NS(self_x_air=f(0.5), self_y=f(-1.0), knockback_x=f(0.25), knockback_y=f(0.75),
                      self_x_ground=f(1.25)) if velocities else None,
    )
    pre = NS(joystick=NS(x=f(0.5), y=f(-0.3)), cstick=NS(x=f(0.0), y=f(1.0)), triggers=f(0.35),
             buttons_physical=f(A | X, np.uint16))
    return NS(leader=NS(pre=pre, post=post))


def fake_game(T=5, stage=0x20, ports=None):
    ports = ports or (fake_port(T), fake_port(T))
    return NS(start=NS(stage=stage), frames=NS(id=np.arange(-123, -123 + T, dtype=np.int32), ports=tuple(ports)))


def test_from_peppi_reads_every_field():
    players, stage, frame_ids = from_peppi(fake_game())
    assert players.dtype == PLAYER_DTYPE and players.shape == (5, 2)
    assert stage == 0 and frame_ids[0] == -123
    p = players[0, 0]
    assert (p["x"], p["y"], p["percent"], p["stocks"]) == (-60.0, 0.0, 12.5, 4)
    assert p["facing"] and p["on_ground"] and not p["invulnerable"]
    assert p["hitlag"] == 2.0 and p["hitstun"] == 10.0  # truncated like libmelee's int()
    assert p["action_frame"] == 3 + state.ZERO_INDEXED[FOX, 14] and p["action"] == 14 and p["character"] == FOX
    assert (p["speed_air_x"], p["speed_y"], p["speed_attack_x"], p["speed_attack_y"], p["speed_ground_x"]) \
        == (0.5, -1.0, 0.25, 0.75, 1.25)
    assert np.isclose(p["main_x"], 0.5) and np.isclose(p["main_y"], -0.3)
    assert (p["c_x"], p["c_y"]) == (0.0, 1.0) and np.isclose(p["trigger"], 0.35)
    assert p["buttons"] == A | X


def test_zero_indexed_actions_get_libmelee_plus_one():
    character, action = 22, 323  # marked zero-indexed in libmelee's actiondata.csv
    assert state.ZERO_INDEXED[character, action]
    players, _, _ = from_peppi(fake_game(ports=(fake_port(5, character=character, action=action), fake_port(5))))
    assert players[0, 0]["action_frame"] == 4.0


def test_non_finite_hitstun_reads_as_zero():
    players, _, _ = from_peppi(fake_game(ports=(fake_port(5, misc_as=np.nan), fake_port(5))))
    assert players[0, 0]["hitstun"] == 0.0


def test_replay_without_speeds_raises_missing_data():
    with pytest.raises(state.MissingData):
        from_peppi(fake_game(ports=(fake_port(5, velocities=False), fake_port(5))))


def test_unknown_stage_maps_to_other():
    _, stage, _ = from_peppi(fake_game(stage=0x99))
    assert stage == state.OTHER_STAGE


def test_libmelee_and_peppi_give_the_same_record():
    gs = melee.GameState()
    gs.stage = melee.Stage.FINAL_DESTINATION
    for port in (1, 2):
        p = melee.PlayerState()
        p.position.x, p.position.y, p.percent, p.stock = -60.0, 0.0, 12.5, 4
        p.facing, p.on_ground, p.invulnerable = True, True, False
        p.hitlag_left, p.hitstun_frames_left, p.jumps_left, p.shield_strength = 2, 10, 2, 60.0
        # libmelee itself adds 1 to the action frame of zero-indexed actions.
        p.action, p.character = melee.Action(14), melee.Character.FOX
        p.action_frame = 3 + int(state.ZERO_INDEXED[FOX, 14])
        p.speed_air_x_self, p.speed_y_self, p.speed_x_attack = 0.5, -1.0, 0.25
        p.speed_y_attack, p.speed_ground_x_self = 0.75, 1.25
        c = p.controller_state
        c.main_stick = (np.float32(0.5) / 2 + 0.5, np.float32(-0.3) / 2 + 0.5)
        c.c_stick, c.l_shoulder = (0.5, 1.0), np.float32(0.35)
        c.button[melee.Button.BUTTON_A] = c.button[melee.Button.BUTTON_X] = True
        gs.players[port] = p
    live = from_libmelee(gs)
    replay = from_peppi(fake_game())[0][0]
    assert state.stage_index(gs.stage) == 0
    for name in PLAYER_DTYPE.names:
        assert np.allclose(live[name].astype(float), replay[name].astype(float), atol=1e-5), name
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_state.py -v`
Expected: FAIL with `ImportError: cannot import name 'state' from 'smash_rl'` (module missing).

- [ ] **Step 4: Implement `smash_rl/state.py`**

```python
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
    """(raw frames (T, n_ports), stage index, frame ids (T,)) of a peppi-py Game."""
    frame_ids = _column(game.frames.id, np.int32)
    players = np.zeros((len(frame_ids), len(game.frames.ports)), dtype=PLAYER_DTYPE)
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
    return players, STAGE_FROM_SLIPPI.get(int(game.start.stage), OTHER_STAGE), frame_ids
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_state.py -v`
Expected: 6 passed.

---

### Task 2: Controller classes (`smash_rl/codec.py`, part 1)

**Files:**
- Create: `smash_rl/codec.py`, `tests/test_codec_controller.py`

**Interfaces:**
- Consumes: `smash_rl.state.BUTTON_BITS`, `PLAYER_DTYPE` (fields `main_x main_y c_x c_y trigger buttons`).
- Produces: `ACTION_SIZES = (48, 17, 17, 5)`, `NEUTRAL = np.array([0, 8, 8, 0])`, `STICK_SEND: np.ndarray (17,)`, `CSTICK_SEND`, `TRIGGER_SEND`, `button_class(a, b, jump, z, trigger) -> int array`, `button_parts(cls) -> (a, b, jump, z, trigger)`, `snap_axis(values) -> int64 array`, `controller_classes(frames) -> int64 array (..., 4)`, `apply_controller(pad, action) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_codec_controller.py`:

```python
import melee
import numpy as np

from smash_rl import codec
from smash_rl.state import BUTTON_BITS, PLAYER_DTYPE

Bt = melee.Button


def frames(n=1, **fields):
    f = np.zeros(n, dtype=PLAYER_DTYPE)
    for name, value in fields.items():
        f[name] = value
    return f


def bits(*buttons):
    return sum(BUTTON_BITS[b] for b in buttons)


def test_neutral_controller_is_neutral_class():
    assert codec.controller_classes(frames())[0].tolist() == codec.NEUTRAL.tolist()


def test_button_classes_cover_all_48_combinations():
    seen = set()
    for a, b, j, z, t in np.ndindex(2, 2, 2, 2, 3):
        cls = int(codec.button_class(a, b, j, z, t))
        assert tuple(int(v) for v in codec.button_parts(cls)) == (a, b, j, z, t)
        seen.add(cls)
    assert seen == set(range(48))


def test_buttons_snap_with_x_or_y_as_jump_and_l_or_r_as_full_trigger():
    f = frames(4, buttons=[bits(Bt.BUTTON_A, Bt.BUTTON_X), bits(Bt.BUTTON_Y), bits(Bt.BUTTON_L), bits(Bt.BUTTON_R)])
    got = codec.controller_classes(f)[:, 0].tolist()
    assert got == [int(codec.button_class(1, 0, 1, 0, 0)), int(codec.button_class(0, 0, 1, 0, 0)),
                   int(codec.button_class(0, 0, 0, 0, 2)), int(codec.button_class(0, 0, 0, 0, 2))]


def test_light_trigger_threshold_and_z_ignores_analog():
    f = frames(3, trigger=[0.35, 0.29, 0.35], buttons=[0, 0, bits(Bt.BUTTON_Z)])
    got = codec.controller_classes(f)[:, 0].tolist()
    assert got == [int(codec.button_class(0, 0, 0, 0, 1)), 0, int(codec.button_class(0, 0, 0, 1, 0))]


def test_stick_snapping():
    got = codec.snap_axis([0.0, 1.0, -1.0, 0.2875, 0.3, 0.5, 0.7071, -0.3]).tolist()
    assert got == [8, 16, 0, 10, 10, 12, 14, 6]


def test_processed_stick_values_never_snap_to_dead_classes():
    # The game reports 0 or a value past the deadzone (multiples of 1/80): ±0.125 never occurs.
    processed = np.concatenate([[0.0], np.arange(23, 81) / 80.0, -np.arange(23, 81) / 80.0])
    assert not np.isin(codec.snap_axis(processed), [7, 9]).any()


def test_sent_stick_values_snap_back_to_their_class():
    for k in range(17):
        if k not in (7, 9):
            assert codec.snap_axis(codec.STICK_SEND[k]) == k


def test_c_stick_directions():
    f = frames(6, c_x=[0, 0, 0, -1, 1, 0.2], c_y=[0, 1, -1, 0, 0, 0.1])
    assert codec.controller_classes(f)[:, 3].tolist() == [0, 1, 2, 3, 4, 0]


class FakePad:
    def __init__(self):
        self.calls = []

    def press_button(self, b):
        self.calls.append(("press", b))

    def release_button(self, b):
        self.calls.append(("release", b))

    def press_shoulder(self, b, amount):
        self.calls.append(("shoulder", b, amount))

    def tilt_analog_unit(self, b, x, y):
        self.calls.append(("tilt", b, float(x), float(y)))


def test_apply_controller_sets_every_input():
    pad = FakePad()
    codec.apply_controller(pad, [int(codec.button_class(1, 0, 0, 0, 2)), 16, 8, 1])
    assert pad.calls == [
        ("press", Bt.BUTTON_A), ("release", Bt.BUTTON_B), ("release", Bt.BUTTON_X), ("release", Bt.BUTTON_Z),
        ("press", Bt.BUTTON_L), ("shoulder", Bt.BUTTON_L, 1.0),
        ("tilt", Bt.BUTTON_MAIN, 1.0, 0.0), ("tilt", Bt.BUTTON_C, 0.0, 1.0),
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_codec_controller.py -v`
Expected: FAIL with `ImportError: cannot import name 'codec'`.

- [ ] **Step 3: Implement the controller part of `smash_rl/codec.py`**

```python
"""Controller classes and model inputs, both computed from raw frames (smash_rl.state).

The agent picks a controller as four classes, in this order (the model conditions
each on the earlier ones): buttons (A, B, jump, Z, trigger level), main stick x,
main stick y, C-stick direction. Replays are snapped to these classes for the
training labels; live, a class tuple is decoded and sent to Dolphin.
"""

import melee
import numpy as np

from smash_rl.state import BUTTON_BITS

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_codec_controller.py -v`
Expected: 9 passed.

---

### Task 3: Dolphin layer split, per-frame env, vectorized envs

**Files:**
- Create: `smash_rl/dolphin.py`, `tests/test_env_queue.py`
- Rewrite: `smash_rl/env.py`, `smash_rl/vec_env.py`, `test_env.py`
- Modify: `requirements.txt` (remove `stable-baselines3==2.3.2`), `README.md`, `AGENTS.md`

**Interfaces:**
- Consumes: `codec.apply_controller`, `codec.ACTION_SIZES`, `codec.NEUTRAL`, `state.from_libmelee`, `state.stage_index`, `state.PLAYER_DTYPE`.
- Produces:
  - `smash_rl.dolphin`: `PROJECT_DIR`, `SYSTEM`, `DEFAULT_DOLPHIN`, `HEADLESS_DOLPHIN`, `CAN_RUN_HEADLESS`, `DEFAULT_ISO`, `INSTANCES_DIR`, `kill_children_on_exit()`, `class Dolphin(port, *, headless, dolphin_path, dolphin_home, iso_path, gfx_backend, gfx_settings, emulation_speed, connect_timeout, human_opponent, instant_restart, save_replays)` with `start()`, `stop()`, `alive() -> bool`, `next_state() -> melee.GameState`, attributes `pad`, `cpu_pad`, `instant_restart`.
  - `smash_rl.env`: `class DelayQueue(delay)` with `reset()` and `push(action) -> np.ndarray`; `class SmashEnv(character="FOX", opponent="FOX", stages=("FINAL_DESTINATION",), cpu_level=9, delay=18, port=51441, headless=False, dolphin_path=None, dolphin_home=None, iso_path=DEFAULT_ISO, gfx_backend="OGL", gfx_settings=None, emulation_speed=0.0, max_menu_frames=6000, connect_timeout=120.0, save_replays=None)`; observations are dicts `{"players": PLAYER_DTYPE (2,), "stage": int, "frame": int}`; `step(action (4,)) -> (obs, 0.0, terminated, False, info)`; info keys `damage_dealt damage_taken kills deaths frames cpu_level opponent` and at game end `won stock_diff`; `step_and_autoreset(env, action) -> (obs, reward, done, info)`; `make_env(rank, base_port, **kw)`; `CloudpickledFn`; `env_worker`.
  - `smash_rl.vec_env`: `make_vec_env(n_envs, base_port, subprocess=True, **env_kwargs)` returning an object with `n`, `reset() -> (obs dict, infos)`, `step(actions (n, 4)) -> (obs dict, rewards (n,), dones (n,), infos)`, `env_method(name, *args)`, `close()`; stacked obs `{"players": (n, 2) PLAYER_DTYPE, "stage": (n,) int64, "frame": (n,) int64}`.

- [ ] **Step 1: Write the failing test for the delay queue**

Create `tests/test_env_queue.py`:

```python
import numpy as np
import pytest

from smash_rl.codec import NEUTRAL
from smash_rl.env import DelayQueue


def test_delay_one_applies_immediately():
    q = DelayQueue(1)
    assert q.push(np.array([5, 1, 2, 3])).tolist() == [5, 1, 2, 3]


def test_decision_takes_effect_delay_frames_later():
    q = DelayQueue(3)
    applied = [q.push(np.array([i, 8, 8, 0]))[0] for i in range(1, 6)]
    # delay 3: the first two frames get neutral input, then each decision lands two pushes later.
    assert applied == [NEUTRAL[0], NEUTRAL[0], 1, 2, 3]


def test_reset_drops_queued_decisions():
    q = DelayQueue(3)
    q.push(np.array([7, 8, 8, 0]))
    q.push(np.array([9, 8, 8, 0]))
    q.reset()
    assert q.push(np.array([1, 8, 8, 0])).tolist() == NEUTRAL.tolist()


def test_delay_must_be_positive():
    with pytest.raises(ValueError):
        DelayQueue(0)
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_env_queue.py -v`
Expected: FAIL with `ImportError: cannot import name 'DelayQueue'`.

- [ ] **Step 2: Create `smash_rl/dolphin.py`**

Move these from today's `smash_rl/env.py` **verbatim**: the module constants `PROJECT_DIR`, `SYSTEM`, `DEFAULT_DOLPHIN`, `HEADLESS_DOLPHIN`, `CAN_RUN_HEADLESS`, `DEFAULT_ISO`, `INSTANCES_DIR`, `WINDOWS_PIPE` (with their comments), and `_link_or_copy`, `_windows_dolphin_copy`, `_Controller`, `_Console`, `_JOB`, `_kill_children_on_exit` (rename it `kill_children_on_exit`, it is now imported by other modules). Then add the `Dolphin` class, built from today's `SmashEnv._start_dolphin`, `_stop_dolphin`, `_next_state` and `_alive`:

```python
"""One Slippi Dolphin process driven through libmelee, with the fixes Windows needs.

(Keep the module docstring short; the Windows details live in the docstrings below.)
"""

import configparser
import os
import platform
import shutil
import time

import melee
from melee.slippstream import SlippstreamClient

# ... moved constants, _link_or_copy, _windows_dolphin_copy, _Controller, _Console,
# _JOB, kill_children_on_exit, exactly as they are in env.py today ...


class Dolphin:
    """One Dolphin: `pad` drives port 1, `cpu_pad` port 2 (None when a person plays port 2).

    save_replays: a directory where Dolphin writes a Slippi replay of every game, or None.
    """

    def __init__(self, port: int, *, headless: bool = False, dolphin_path: str = None,
                 dolphin_home: str = None, iso_path: str = DEFAULT_ISO, gfx_backend: str = "OGL",
                 gfx_settings: dict = None, emulation_speed: float = 0.0, connect_timeout: float = 120.0,
                 human_opponent: bool = False, instant_restart: bool = False, save_replays: str = None):
        if headless and not CAN_RUN_HEADLESS:
            raise ValueError(f"headless mode needs Linux (the ExiAI Dolphin is a Linux AppImage), not {SYSTEM}")
        self.port = port
        self.headless = headless
        self.dolphin_path = dolphin_path or (HEADLESS_DOLPHIN if headless else DEFAULT_DOLPHIN)
        self.dolphin_home = dolphin_home  # None: a fresh temporary Dolphin user folder per start
        self.iso_path = iso_path
        self.gfx_backend = gfx_backend
        self.gfx_settings = gfx_settings or {}  # {"Section": {"Key": value}} for Dolphin's GFX.ini
        self.emulation_speed = emulation_speed
        self.connect_timeout = connect_timeout
        self.human_opponent = human_opponent
        self.instant_restart = instant_restart
        self.save_replays = save_replays
        for path in (self.dolphin_path, self.iso_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found - see setup.ps1 (Windows) or setup.sh")
        self.console = self.pad = self.cpu_pad = None

    def start(self):
        self.stop()
        kill_children_on_exit()
        if self.save_replays:
            os.makedirs(self.save_replays, exist_ok=True)
        dolphin_path, pipe_prefix = self.dolphin_path, None
        if SYSTEM == "Windows":
            dolphin_path, pipe_prefix = _windows_dolphin_copy(self.dolphin_path, self.port)
        console = _Console(
            pipe_prefix=pipe_prefix,
            path=dolphin_path,
            dolphin_home_path=self.dolphin_home,
            tmp_home_directory=self.dolphin_home is None,
            slippi_port=self.port,
            # Blocking input makes Dolphin wait for the agent every frame, which would freeze
            # a pause by a person (no game states arrive to answer with while paused).
            blocking_input=not self.human_opponent,
            fullscreen=False,
            disable_audio=not self.human_opponent,
            save_replays=bool(self.save_replays),
            replay_dir=self.save_replays,
            gfx_backend="Null" if self.headless else self.gfx_backend,
            log_types=[],
            instant_match_restart=self.instant_restart,
            use_exi_inputs=self.headless,
            enable_ffw=self.headless,
        )
        # From here to the end of the method: today's _start_dolphin body from
        # `game_ini = ...` through the pad connect loop, with `self.console`,
        # `self.pad`, `self.cpu_pad` assigned on this object, `self._alive()` replaced
        # by `self.alive()` and `self._stop_dolphin()` by `self.stop()`. Drop the two
        # trailing lines that set `_running_level` and `_game_over` (they stay in the env).

    def stop(self):
        if self.console is not None:
            try:
                self.console.stop()
            except Exception:
                pass
            self.console = None

    def alive(self) -> bool:
        proc = getattr(self.console, "_process", None)
        return self.console is not None and proc is not None and proc.poll() is None

    def next_state(self) -> melee.GameState:
        for _ in range(1000):
            gs = self.console.step()
            if gs is not None:
                return gs
        raise RuntimeError("dolphin stopped sending game states")
```

The `_start_dolphin` body being moved is in today's `smash_rl/env.py` lines 285-337 (`game_ini = ...` through the pad `connect()` loop); copy it unchanged apart from the renames listed in the comment above.

- [ ] **Step 3: Rewrite `smash_rl/env.py`**

```python
"""Super Smash Bros. Melee environment: one Dolphin, one decision per frame.

The agent controls port 1. Each step takes one controller as class indices
(smash_rl.codec) and advances one frame; a decision takes effect `delay` frames
after the frame it was made on, like a human's reaction time. Observations are
raw frames (smash_rl.state) of [agent, opponent] plus the stage index and frame
number; smash_rl.codec turns them into model inputs.

One episode is one full game. Between games the env walks the menus itself; with
one stage and a fixed CPU (windowed) a finished game restarts instantly instead.
The reward is 0 until Phase 2 defines one.
"""

import collections
import itertools
import signal
import weakref

import cloudpickle
import gymnasium as gym
import melee
import numpy as np

from smash_rl import codec
from smash_rl.dolphin import DEFAULT_ISO, Dolphin
from smash_rl.state import from_libmelee, stage_index

IN_GAME = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)
# Opponents for opponent="RANDOM": everyone on the character select screen. The CPU
# can't be made to pick Sheik, but a CPU Zelda may transform into her mid-game.
RANDOM_OPPONENTS = tuple(c for c in melee.Character
                         if c.value < 27 and c not in (melee.Character.SHEIK, melee.Character.NANA))
_LIVE_ENVS = weakref.WeakSet()  # envs in this process, closed when a worker exits


class DelayQueue:
    """Applies each decision `delay` frames after the frame it was made on (delay >= 1)."""

    def __init__(self, delay: int):
        if delay < 1:
            raise ValueError(f"delay must be at least 1 frame, got {delay}")
        self.delay = delay
        self.reset()

    def reset(self):
        self._queue = collections.deque(codec.NEUTRAL.copy() for _ in range(self.delay - 1))

    def push(self, action) -> np.ndarray:
        self._queue.append(np.asarray(action, dtype=np.int64))
        return self._queue.popleft()


class SmashEnv(gym.Env):
    """Agent (port 1) vs a CPU or a person (port 2), one frame per step.

    opponent: a character name for a CPU at cpu_level, "RANDOM" for a new CPU character
    every game, or "HUMAN" for a person with a Dolphin-mapped controller on port 2.
    stages: one or more stage names; with several, each game picks one at random.
    save_replays: a directory where Dolphin saves a Slippi replay of every game.
    """

    metadata = {"render_modes": []}

    def __init__(self, character: str = "FOX", opponent: str = "FOX", stages=("FINAL_DESTINATION",),
                 cpu_level: int = 9, delay: int = 18, port: int = 51441, headless: bool = False,
                 dolphin_path: str = None, dolphin_home: str = None, iso_path: str = DEFAULT_ISO,
                 gfx_backend: str = "OGL", gfx_settings: dict = None, emulation_speed: float = 0.0,
                 max_menu_frames: int = 6000, connect_timeout: float = 120.0, save_replays: str = None):
        super().__init__()
        self.character = melee.Character[character.upper()]
        self.random_opponent = opponent.upper() == "RANDOM"
        self.human_opponent = opponent.upper() == "HUMAN"
        self.opponent = None if self.random_opponent or self.human_opponent else melee.Character[opponent.upper()]
        self.stages = tuple(melee.Stage[s.upper()] for s in ([stages] if isinstance(stages, str) else stages))
        self.stage = self.stages[0]
        self.cpu_level = int(cpu_level)
        self.port = port
        self.max_menu_frames = max_menu_frames
        # Instant Match restarts a finished game without the menus and keeps the stage, so it
        # needs a fixed CPU and one stage; headless never uses it (it crashes the ExiAI build).
        instant_restart = not (self.random_opponent or self.human_opponent or headless) and len(self.stages) == 1
        self.dolphin = Dolphin(
            port, headless=headless, dolphin_path=dolphin_path, dolphin_home=dolphin_home, iso_path=iso_path,
            gfx_backend=gfx_backend, gfx_settings=gfx_settings, emulation_speed=emulation_speed,
            connect_timeout=connect_timeout, human_opponent=self.human_opponent,
            instant_restart=instant_restart, save_replays=save_replays)
        self.queue = DelayQueue(delay)
        self.action_space = gym.spaces.MultiDiscrete(codec.ACTION_SIZES)
        _LIVE_ENVS.add(self)
        self._gs = None
        self._game_over = False
        self._running_level = None
        self._stats = {}
        self._frames = 0

    def set_cpu_level(self, level: int):
        """Takes effect from the next game (restarting Dolphin when games restart instantly)."""
        self.cpu_level = int(np.clip(level, 1, 9))

    def _go_to_game(self):
        """Navigate menus until a fresh game with the right setup has started."""
        pad, cpu_pad = self.dolphin.pad, self.dolphin.cpu_pad
        need_new_game = self._game_over
        last_frame = self._gs.frame if (need_new_game and self._gs is not None) else None
        # MenuHelper remembers that it already picked a stage, so use fresh ones for every visit.
        menu, cpu_menu = melee.MenuHelper(), melee.MenuHelper()
        # A person can take as long as they like on the character select screen.
        for _ in itertools.count() if self.human_opponent else range(self.max_menu_frames):
            gs = self.dolphin.next_state()
            if gs.menu_state in IN_GAME:
                pad.release_all()
                if cpu_pad is not None:
                    cpu_pad.release_all()
                if need_new_game:
                    # Wait for the previous game to end and the next one to begin.
                    if gs.frame < 0 or (last_frame is not None and gs.frame < last_frame):
                        need_new_game = False
                    last_frame = gs.frame
                    continue
                p1, p2 = gs.players.get(1), gs.players.get(2)
                if p1 is not None and p2 is not None and p1.stock > 0 and p2.stock > 0:
                    return gs
                continue
            need_new_game = False
            cpu_ready = False
            if cpu_pad is not None and gs.menu_state == melee.Menu.CHARACTER_SELECT:
                p2 = gs.players.get(2)
                cpu_ready = p2 is not None and p2.cpu_level == self.cpu_level and p2.character == self.opponent
                cpu_menu.menu_helper_simple(gs, cpu_pad, self.opponent, self.stage,
                                            cpu_level=self.cpu_level, autostart=False)
            elif cpu_pad is not None:
                cpu_pad.release_all()
            # Only start once the CPU slot is set up (a person on port 2 presses Start
            # themselves); autostart also makes port 1 pick the stage.
            menu.menu_helper_simple(gs, pad, self.character, self.stage, cpu_level=0,
                                    autostart=gs.menu_state != melee.Menu.CHARACTER_SELECT or cpu_ready)
        raise RuntimeError("stuck in menus")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_opponent:
            self.opponent = RANDOM_OPPONENTS[self.np_random.integers(len(RANDOM_OPPONENTS))]
        if len(self.stages) > 1:
            self.stage = self.stages[self.np_random.integers(len(self.stages))]
        for attempt in range(3):
            try:
                if self.dolphin.alive() and self._running_level != self.cpu_level and self.dolphin.instant_restart:
                    self.dolphin.stop()
                if not self.dolphin.alive():
                    self.dolphin.start()
                    self._running_level = self.cpu_level
                    self._game_over = False
                gs = self._go_to_game()
                break
            except Exception as e:  # dolphin crashed or hung in menus: restart it
                if self.human_opponent:
                    raise  # most likely the person closed the Dolphin window
                print(f"[port {self.port}] restarting dolphin after: {e}")
                self.dolphin.stop()
        else:
            raise RuntimeError("could not start a game after 3 attempts")
        if self.human_opponent:
            self.opponent = gs.players[2].character
        self._gs = gs
        self._game_over = False
        self._frames = 0
        self.queue.reset()
        self._prev = self._snapshot(gs)
        self._stats = {"damage_dealt": 0.0, "damage_taken": 0.0, "kills": 0, "deaths": 0}
        return self._obs(gs), self._info(gs, done=False)

    def step(self, action):
        codec.apply_controller(self.dolphin.pad, self.queue.push(action))
        gs = self.dolphin.next_state()
        self._frames += 1
        # Leaving the game, a player disappearing or the frame counter going back (an
        # instantly restarted game) all end the episode; the last in-game state is kept.
        in_game = (gs.menu_state in IN_GAME and gs.frame >= self._gs.frame
                   and 1 in gs.players and 2 in gs.players)
        if in_game:
            self._track(gs)
            self._gs = gs
        players = self._gs.players
        terminated = not in_game or players[1].stock == 0 or players[2].stock == 0
        self._game_over = terminated
        return self._obs(self._gs), 0.0, terminated, False, self._info(self._gs, done=terminated)

    @staticmethod
    def _snapshot(gs):
        p1, p2 = gs.players[1], gs.players[2]
        return float(p1.percent), int(p1.stock), float(p2.percent), int(p2.stock)

    def _track(self, gs):
        my_pct, my_stock, op_pct, op_stock = self._snapshot(gs)
        prev_my_pct, prev_my_stock, prev_op_pct, prev_op_stock = self._prev
        self._prev = (my_pct, my_stock, op_pct, op_stock)
        deaths = max(0, prev_my_stock - my_stock)
        kills = max(0, prev_op_stock - op_stock)
        # Percent resets to 0 on death, so only count increases within a stock.
        self._stats["damage_taken"] += max(0.0, my_pct - prev_my_pct) if not deaths else 0.0
        self._stats["damage_dealt"] += max(0.0, op_pct - prev_op_pct) if not kills else 0.0
        self._stats["kills"] += kills
        self._stats["deaths"] += deaths

    @staticmethod
    def _obs(gs) -> dict:
        return {"players": from_libmelee(gs), "stage": stage_index(gs.stage), "frame": int(gs.frame)}

    def _info(self, gs, done: bool) -> dict:
        info = dict(self._stats, frames=self._frames, cpu_level=self.cpu_level,
                    opponent=getattr(self.opponent, "name", str(self.opponent)))
        if done:
            me, op = gs.players[1], gs.players[2]
            info["won"] = float(op.stock < me.stock or (op.stock == me.stock and op.percent > me.percent))
            info["stock_diff"] = int(me.stock) - int(op.stock)
        return info

    def close(self):
        self.dolphin.stop()


def step_and_autoreset(env, action):
    """One step; a finished game continues with the next game's first observation
    (info["final_observation"] keeps the finished game's last one)."""
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        info["final_observation"] = obs
        obs, _ = env.reset()
    return obs, reward, done, info


def make_env(rank: int = 0, base_port: int = 51441, **env_kwargs):
    """Return a thunk that builds one Smash env with its own Dolphin port."""

    def _init():
        return SmashEnv(port=base_port + rank, **env_kwargs)

    return _init


class CloudpickledFn:
    """Sends a function (lambdas and closures too) to a worker with cloudpickle."""

    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, data):
        self.fn = cloudpickle.loads(data)


def _exit_on_signal(signum, frame):
    raise SystemExit(1)


def env_worker(remote, parent_remote, env_fn: CloudpickledFn):
    """Worker process of smash_rl.vec_env: runs one env, closes its Dolphin on exit.

    Leaves Ctrl+C / hangups to the main process. Imports neither PyTorch nor SB3, so
    each worker stays small (~50 MB).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGHUP"):  # not on Windows
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _exit_on_signal)
    parent_remote.close()
    try:
        env = env_fn.fn()
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break
            if cmd == "step":
                remote.send(step_and_autoreset(env, data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "env_method":
                remote.send(getattr(env, data[0])(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    finally:
        for live_env in list(_LIVE_ENVS):
            live_env.close()
```

- [ ] **Step 4: Rewrite `smash_rl/vec_env.py` without Stable-Baselines3**

```python
"""Many SmashEnvs stepped in lockstep: one worker process and one Dolphin each.

Workers run smash_rl.env.env_worker, which imports neither PyTorch nor SB3, so each
costs ~50 MB; the main process holds the model and acts for all games at once.
"""

import contextlib
import multiprocessing as mp
import os
import sys
import types

import numpy as np

from smash_rl.dolphin import kill_children_on_exit
from smash_rl.env import CloudpickledFn, env_worker, make_env, step_and_autoreset


@contextlib.contextmanager
def _lean_spawn():
    """Start spawned processes without the main script and with single-threaded BLAS.

    A spawned process first re-imports the main script (as __mp_main__), so a training
    script's `import torch` would land in every worker; hiding __main__ while the workers
    start skips that (they only need smash_rl.env). numpy's OpenBLAS also commits memory
    for one buffer per CPU thread in every process, which the workers don't need.
    """
    main = sys.modules["__main__"]
    blas = os.environ.get("OPENBLAS_NUM_THREADS")
    sys.modules["__main__"] = types.ModuleType("__main__")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        sys.modules["__main__"] = main
        if blas is None:
            del os.environ["OPENBLAS_NUM_THREADS"]
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = blas


def stack_obs(observations) -> dict:
    return {
        "players": np.stack([o["players"] for o in observations]),
        "stage": np.array([o["stage"] for o in observations], dtype=np.int64),
        "frame": np.array([o["frame"] for o in observations], dtype=np.int64),
    }


def _stack_steps(results):
    obs, rewards, dones, infos = zip(*results)
    return stack_obs(obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), list(infos)


class SubprocVecEnv:
    """Envs in non-daemon worker processes (libmelee starts a helper process per env,
    and daemon processes may not have children)."""

    def __init__(self, env_fns, start_method: str = "spawn"):
        self.n = len(env_fns)
        kill_children_on_exit()  # Windows: workers (and their Dolphins) die with the main process
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.processes = []
        with _lean_spawn():
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
                process = ctx.Process(target=env_worker, args=(work_remote, remote, CloudpickledFn(env_fn)),
                                      daemon=False)
                process.start()
                self.processes.append(process)
                work_remote.close()
        self.closed = False

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        return stack_obs([obs for obs, _ in results]), [info for _, info in results]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", np.asarray(action)))
        return _stack_steps([remote.recv() for remote in self.remotes])

    def env_method(self, name, *args, **kwargs) -> list:
        for remote in self.remotes:
            remote.send(("env_method", (name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join()
        self.closed = True


class LocalVecEnv:
    """The same interface with the envs in this process (one game, or debugging)."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)

    def reset(self):
        results = [env.reset() for env in self.envs]
        return stack_obs([obs for obs, _ in results]), [info for _, info in results]

    def step(self, actions):
        return _stack_steps([step_and_autoreset(env, a) for env, a in zip(self.envs, actions)])

    def env_method(self, name, *args, **kwargs) -> list:
        return [getattr(env, name)(*args, **kwargs) for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()


def make_vec_env(n_envs: int = 4, base_port: int = 51441, subprocess: bool = True, **env_kwargs):
    """n games (one Dolphin each) on Slippi ports base_port .. base_port + n - 1."""
    thunks = [make_env(i, base_port, **env_kwargs) for i in range(n_envs)]
    if subprocess and n_envs > 1:
        return SubprocVecEnv(thunks)
    return LocalVecEnv(thunks)
```

Remove the line `stable-baselines3==2.3.2` from `requirements.txt` (the env no longer uses SB3; Phase 2 writes its own PPO).

- [ ] **Step 5: Run the unit tests**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests pass (state, controller and queue tests).

- [ ] **Step 6: Rewrite `test_env.py` for per-frame steps**

```python
"""Sanity check for the per-frame env: start Dolphin, check inputs arrive, measure speed.

Usage:
    python test_env.py                       # one game window
    python test_env.py --n-envs 10           # 10 games in parallel, as in training
    python test_env.py --stage BATTLEFIELD --cpu-level 9
"""

import argparse
import time

import numpy as np

from smash_rl.codec import ACTION_SIZES
from smash_rl.state import PLAYER_DTYPE
from smash_rl.vec_env import make_vec_env


def main():
    p = argparse.ArgumentParser(description="Check the per-frame env and measure its speed")
    p.add_argument("--n-envs", type=int, default=1, help="games in parallel, one worker process and Dolphin each")
    p.add_argument("--frames", type=int, default=3000, help="random-controller frames per game")
    p.add_argument("--cpu-level", type=int, default=3)
    p.add_argument("--opponent", default="FOX", help="CPU character, or RANDOM")
    p.add_argument("--stage", default="FINAL_DESTINATION", help="stage name, e.g. BATTLEFIELD")
    p.add_argument("--delay", type=int, default=18, help="frames between a decision and its input")
    p.add_argument("--port", type=int, default=51441, help="Slippi port of the first game")
    p.add_argument("--headless", action="store_true", help="no window and fast-forward mode (Linux only)")
    args = p.parse_args()

    n = args.n_envs
    print(f"starting {n} dolphin(s) on ports {args.port}-{args.port + n - 1} and navigating menus...")
    t = time.time()
    env = make_vec_env(n_envs=n, base_port=args.port, opponent=args.opponent, cpu_level=args.cpu_level,
                       stages=(args.stage,), delay=args.delay, headless=args.headless)
    try:
        obs, _ = env.reset()
        assert obs["players"].dtype == PLAYER_DTYPE and obs["players"].shape == (n, 2)
        print(f"in game after {time.time() - t:.1f}s | first frame {obs['frame'].tolist()} | stage {obs['stage'].tolist()}")
        # Hold the stick right: every agent must move right once the delay has passed.
        x0 = obs["players"]["x"][:, 0].copy()
        for _ in range(args.delay + 60):
            obs, _, _, _ = env.step(np.tile([0, 16, 8, 0], (n, 1)))
        stuck = [i for i in range(n) if obs["players"]["x"][i, 0] - x0[i] < 20]
        print(f"inputs: {n - len(stuck)}/{n} agents moved right" + (f", not moving: games {stuck}" if stuck else ""))
        rng = np.random.default_rng(0)
        start, games = time.time(), 0
        for _ in range(args.frames):
            actions = np.stack([rng.integers(0, size, n) for size in ACTION_SIZES], axis=1)
            obs, _, dones, _ = env.step(actions)
            games += int(dones.sum())
        elapsed = time.time() - start
        print(f"random controllers: {n * args.frames} frames, {games} finished games | "
              f"{n * args.frames / elapsed:.0f} frames/s in total, {args.frames / elapsed:.0f} per game "
              f"(real time is 60)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Live check, one game**

Run: `.venv\Scripts\python.exe test_env.py --frames 1500`
Expected: a Dolphin window opens; output shows `in game after <10s`, `inputs: 1/1 agents moved right`, and a frames/s line; the process exits with no `Slippi Dolphin.exe` left (`tasklist | findstr /i slippi` prints nothing).

- [ ] **Step 8: Live check, parallel speed (input for Phase 2's game count)**

Run: `.venv\Scripts\python.exe test_env.py --n-envs 10 --frames 1500 --port 51600`, then the same with `--n-envs 20`.
Expected: `inputs: 10/10` and `20/20 agents moved right`. Record the two `frames/s in total` numbers in README's "Parallel games" table (replace the old frame-skip numbers). If 20 games is below ~600 frames/s in total, note in README that Phase 2 should start at 10 games.

- [ ] **Step 9: Update README.md and AGENTS.md**

In README: replace the "The environment" section with the per-frame description (observation = raw frames + stage + frame; action = 4 controller classes; `delay` default 18; `stages`; `save_replays`), list `test_env.py` flags (`--n-envs --frames --cpu-level --opponent --stage --delay --port --headless`), and replace `smash_rl/vec_env.py`'s description ("SB3") with "lockstep vectorized envs, no SB3". In AGENTS.md's map: add `smash_rl/dolphin.py`, `smash_rl/state.py`, `smash_rl/codec.py`; describe `env.py` as per-frame with `DelayQueue`; remove the 898-float observation section and `ACTIONS`, and replace them with a pointer to `state.PLAYER_DTYPE` and `codec.ACTION_SIZES`.

---

### Task 4: Live consistency checks

**Files:**
- Create: `tests/live/check_controller_roundtrip.py`, `tests/live/check_replay_vs_live.py`

**Interfaces:**
- Consumes: `SmashEnv(delay=1, save_replays=...)`, `codec.controller_classes`, `codec.STICK_SEND`, `state.from_peppi`, `peppi_py.read_slippi`.
- Produces: two scripts that exit 0 on success and print mismatches otherwise. If either fails, fix `state.py` / `codec.py` (and their unit tests) until both pass; Task 5 must not start before.

- [ ] **Step 1: Write the controller round-trip check**

Create `tests/live/check_controller_roundtrip.py`:

```python
"""Live check: every controller class sent through the pipe reads back as the same class.

Plays a game against a level 1 CPU with delay 1, so the decision made after frame t is
the input of frame t+1, and snaps the controller the game recorded for that frame.
Stick classes outside the stick's circle (the game clamps them) and the ±0.125 classes
(inside the deadzone, never produced by replays) are not expected to round-trip.

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
```

- [ ] **Step 2: Run it**

Run: `.venv\Scripts\python.exe tests/live/check_controller_roundtrip.py`
Expected: `<N>/<N> controller classes read back unchanged` and exit code 0. If some classes fail, the printed raw values show what the game recorded: adjust `LIGHT_TRIGGER`, `TRIGGER_SEND`, `CSTICK_THRESHOLD` or `STICK_SEND` in `codec.py` (and the matching unit test expectations) and re-run until it passes.

- [ ] **Step 3: Write the replay-vs-live check**

Create `tests/live/check_replay_vs_live.py`:

```python
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
        time.sleep(2)  # let Slippi finish writing the replay
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
```

- [ ] **Step 4: Run it**

Run: `.venv\Scripts\python.exe tests/live/check_replay_vs_live.py`
Expected: `all fields match` and exit code 0 (several thousand common frames). If a field differs, fix the conversion in `state.py` (`from_peppi` or `from_libmelee`) so both follow libmelee's definition, update `tests/test_state.py` to pin the corrected behavior, run `pytest`, and re-run this check until it passes.

---

### Task 5: Replay dataset preparation (`prepare_data.py`)

**Files:**
- Create: `smash_rl/replays.py`, `prepare_data.py`, `tests/test_replays.py`
- Modify: `requirements.txt` (add `huggingface_hub`), `README.md`

**Interfaces:**
- Consumes: `state.from_peppi`, `state.MissingData`, `state.STAGE_FROM_SLIPPI`, `peppi_py.read_slippi`.
- Produces: `class Skip(Exception)`, `check_game(game, character: int) -> list[int]` (slots of the character), `game_split(game_id: str) -> str`, `convert_safely(slp_bytes: bytes, game_id: str, out_path: str, character: int) -> dict` (an index entry, or `{"skip": reason}`), `read_index(path) -> list[dict]`. On disk: `<data>/parsed/<archive>/<game_id>.npz` with arrays `players (T, 2) PLAYER_DTYPE`, `stage` (int8), `first_frame` (int32); `<data>/index.jsonl` lines with keys `game_id archive rank path stage characters slots frames split`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replays.py`:

```python
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from smash_rl import replays
from smash_rl.state import FOX


def game(players=2, human=True, stage=0x20, end=2, play_frames=2000, characters=(FOX, 9)):
    ports = tuple(NS(leader=NS(post=NS(character=np.full(3, c, np.uint8)))) for c in characters[:players])
    return NS(
        start=NS(players=[NS(type=0 if human else 1) for _ in range(players)], stage=stage),
        end=None if end is None else NS(method=end),
        frames=NS(id=np.arange(-123, play_frames, dtype=np.int32), ports=ports),
    )


def test_accepts_a_normal_fox_game():
    assert replays.check_game(game(), FOX) == [0]


def test_fox_ditto_gives_both_slots():
    assert replays.check_game(game(characters=(FOX, FOX)), FOX) == [0, 1]


@pytest.mark.parametrize("kwargs, reason", [
    (dict(players=3, characters=(FOX, 9, 9)), "not 2 players"),
    (dict(human=False), "not human"),
    (dict(stage=0x99), "stage"),
    (dict(end=7), "quit or disconnect"),
    (dict(end=None), "quit or disconnect"),
    (dict(play_frames=1000), "short"),
    (dict(characters=(9, 9)), "character absent"),
])
def test_rejects(kwargs, reason):
    with pytest.raises(replays.Skip, match=reason):
        replays.check_game(game(**kwargs), FOX)


def test_split_is_deterministic_and_about_two_percent():
    splits = [replays.game_split(f"g{i}") for i in range(20000)]
    assert splits == [replays.game_split(f"g{i}") for i in range(20000)]
    assert 0.01 < splits.count("val") / len(splits) < 0.03


def test_corrupt_replay_is_skipped(tmp_path):
    out = tmp_path / "x.npz"
    assert replays.convert_safely(b"not a replay", "x", str(out), FOX) == {"skip": "parse error"}
    assert not out.exists() and not list(tmp_path.iterdir())


def test_replay_without_speeds_is_skipped(tmp_path, monkeypatch):
    g = game()
    for port in g.frames.ports:
        port.leader.post.velocities = None
        port.leader.pre = None
    monkeypatch.setattr(replays.peppi_py, "read_slippi", lambda path: g)
    assert replays.convert_safely(b"x", "g", str(tmp_path / "g.npz"), FOX) == {"skip": "missing data"}
    assert not list(tmp_path.iterdir())


def test_read_index_drops_a_half_written_last_line(tmp_path):
    path = tmp_path / "index.jsonl"
    path.write_text(json.dumps({"game_id": "a"}) + "\n" + '{"game_id": "b", "fra')
    assert [e["game_id"] for e in replays.read_index(str(path))] == ["a"]
    assert replays.read_index(str(tmp_path / "missing.jsonl")) == []
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_replays.py -v`
Expected: FAIL with `ImportError: cannot import name 'replays'`.

- [ ] **Step 2: Implement `smash_rl/replays.py`**

```python
"""Ranked replays (erickfm/melee-ranked-replays): filtering and conversion to raw frames."""

import hashlib
import json
import os

import numpy as np
import peppi_py

from smash_rl.state import STAGE_FROM_SLIPPI, MissingData, from_peppi

REPO = "erickfm/melee-ranked-replays"
MIN_PLAY_FRAMES = 1800  # 30 s after the countdown
NORMAL_ENDS = (1, 2, 3)  # peppi EndMethod TIME, GAME, RESOLVED (not UNRESOLVED or NO_CONTEST)


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
    if game.end is None or int(game.end.method) not in NORMAL_ENDS:
        raise Skip("quit or disconnect")
    if int((np.asarray(game.frames.id) >= 0).sum()) < MIN_PLAY_FRAMES:
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
    np.savez_compressed(out_path, players=players, stage=np.int8(stage), first_frame=np.int32(frame_ids[0]))
    return {"game_id": game_id, "stage": int(stage), "characters": [int(c) for c in players[0]["character"]],
            "slots": slots, "frames": int(len(players)), "split": game_split(game_id)}


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
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_replays.py -v`
Expected: 13 passed.

- [ ] **Step 3: Implement `prepare_data.py`**

```python
"""Download Fox (or another character's) ranked replays and convert them to raw frames.

Processes one archive at a time: download it, parse every replay on all cores, write
one .npz per kept game and a line per game to index.jsonl, then delete the archive.
Rerunning skips finished archives and games already in the index.

Usage:
    python prepare_data.py                                   # Fox, master-master + master-diamond, all 6 archives
    python prepare_data.py --ranks master-master --archives 1 2
    python prepare_data.py --character LINK --ranks master-diamond --archives 3 --data-dir D:\\smash_smoke
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import tarfile
import time

import melee
from huggingface_hub import hf_hub_download

from smash_rl.replays import REPO, convert_safely, read_index


def parse_args():
    p = argparse.ArgumentParser(description="Download and parse ranked Melee replays")
    p.add_argument("--data-dir", default=r"D:\smash_data", help="where archives, parsed games and index.jsonl go")
    p.add_argument("--character", default="FOX", help="dataset folder and libmelee character name, e.g. FOX, FALCO")
    p.add_argument("--ranks", nargs="+", default=["master-master", "master-diamond"], help="rank pairs to download")
    p.add_argument("--archives", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="source archives a1..a6")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parsing processes")
    p.add_argument("--keep-archives", action="store_true", help="don't delete archives after parsing")
    p.add_argument("--max-games", type=int, default=None, help="stop each archive after this many replays (testing)")
    return p.parse_args()


def process_archive(tar_path, name, rank, args, character, index_file, known):
    out_dir = os.path.join(args.data_dir, "parsed", name)
    os.makedirs(out_dir, exist_ok=True)
    counts, pending = collections.Counter(), {}

    def collect(done):
        for future in done:
            game_id = pending.pop(future)
            result = future.result()
            if "skip" in result:
                counts[result["skip"]] += 1
                continue
            entry = dict(result, archive=name, rank=rank, path=f"parsed/{name}/{game_id}.npz")
            index_file.write(json.dumps(entry) + "\n")
            index_file.flush()
            known.add(game_id)
            counts["kept"] += 1

    with cf.ProcessPoolExecutor(args.workers) as pool, tarfile.open(tar_path, "r:gz") as tar:
        seen = 0
        for member in tar:
            if not member.isfile() or not member.name.endswith(".slp"):
                continue
            if args.max_games is not None and seen >= args.max_games:
                break
            seen += 1
            game_id = os.path.splitext(os.path.basename(member.name))[0]
            if game_id in known:
                counts["already parsed"] += 1
                continue
            data = tar.extractfile(member).read()
            out_path = os.path.join(out_dir, game_id + ".npz")
            pending[pool.submit(convert_safely, data, game_id, out_path, character)] = game_id
            if len(pending) >= 4 * args.workers:
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                collect(done)
        collect(cf.wait(pending)[0])
    return counts


def main():
    args = parse_args()
    character = melee.Character[args.character.upper()].value
    for sub in ("raw", "parsed", "done"):
        os.makedirs(os.path.join(args.data_dir, sub), exist_ok=True)
    index_path = os.path.join(args.data_dir, "index.jsonl")
    entries = read_index(index_path)
    known = {e["game_id"] for e in entries}
    with open(index_path, "w") as f:  # drops a half-written last line from an interrupted run
        f.writelines(json.dumps(e) + "\n" for e in entries)
    with open(index_path, "a") as index_file:
        for rank in args.ranks:
            for archive in args.archives:
                name = f"{args.character.upper()}_{rank}_a{archive}"
                marker = os.path.join(args.data_dir, "done", name)
                if os.path.exists(marker):
                    print(f"{name}: done already")
                    continue
                t = time.time()
                print(f"{name}: downloading...", flush=True)
                tar_path = hf_hub_download(REPO, f"{args.character.upper()}/{name}.tar.gz", repo_type="dataset",
                                           local_dir=os.path.join(args.data_dir, "raw"))
                print(f"{name}: parsing with {args.workers} workers...", flush=True)
                counts = process_archive(tar_path, name, rank, args, character, index_file, known)
                print(f"{name}: {dict(counts)} in {(time.time() - t) / 60:.1f} min", flush=True)
                if args.max_games is None:
                    open(marker, "w").close()
                    if not args.keep_archives:
                        os.remove(tar_path)
    print(f"index: {len(read_index(index_path))} games in {index_path}")


if __name__ == "__main__":
    main()
```

Append `huggingface_hub` to `requirements.txt` under "# Replays and tests", then run `.venv\Scripts\python.exe -m pip install huggingface_hub`.

- [ ] **Step 4: Smoke run on a small archive**

Run: `.venv\Scripts\python.exe prepare_data.py --character LINK --ranks master-diamond --archives 3 --data-dir D:\smash_smoke --workers 8`
Expected: prints `LINK_master-diamond_a3: {'kept': N, ...}` with N between 40 and 84, and `index: N games`. Check one file:

```powershell
.venv\Scripts\python.exe -c "import numpy as np, glob; f = glob.glob(r'D:\smash_smoke\parsed\*\*.npz')[0]; d = np.load(f); print(d['players'].shape, d['players'].dtype.names[:3], int(d['stage']), int(d['first_frame']))"
```

Expected: `(T, 2) ('x', 'y', 'percent') <0-5> -123` with T above 1900. Run the same command again: it prints `done already`.

- [ ] **Step 5: Start the full Fox run in the background**

Run (leave it running; it takes ~2-3 h): `.venv\Scripts\python.exe prepare_data.py --data-dir D:\smash_data`
Expected after the first archive: `FOX_master-master_a1: {'kept': ..., ...}` with most games kept and skip reasons counted. Later tasks can start while it runs; the index grows as games are parsed.

- [ ] **Step 6: Document it in README.md**

Add a "Data" section: what is downloaded (Fox, master-master + master-diamond, ~84k games, ~93 GB compressed, ~50 GB parsed), the filters, the on-disk layout (`parsed/`, `index.jsonl`, `done/`), resuming, and the flags (`--data-dir --character --ranks --archives --workers --keep-archives --max-games`).

---

### Task 6: Model inputs (`smash_rl/codec.py`, part 2)

**Files:**
- Modify: `smash_rl/codec.py`
- Create: `tests/test_codec_features.py`

**Interfaces:**
- Consumes: `PLAYER_DTYPE`, `NUM_STAGES` from `smash_rl.state`.
- Produces: `NUM_ACTIONS = 400`, `NUM_CHARACTERS = 33`, `NUM_JUMPS = 8`, `NUMERIC_PER_PLAYER = 16`, `encode_state(players, stage) -> dict` with keys `numeric (..., 32) float32`, `action (..., 2) int64`, `character (..., 2) int64`, `jumps (..., 2) int64`, `stage (...) int64`, where `players` is `(..., 2)` PLAYER_DTYPE with the agent first and `stage` broadcasts to `players.shape[:-1]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_codec_features.py`:

```python
import numpy as np

from smash_rl import codec
from smash_rl.state import PLAYER_DTYPE


def players(shape=(3,)):
    p = np.zeros(shape + (2,), dtype=PLAYER_DTYPE)
    p["x"][..., 0], p["x"][..., 1] = -60.0, 60.0
    p["percent"][..., 1] = 50.0
    p["stocks"] = 4
    p["facing"][..., 0] = True
    p["action"][..., 1] = 500
    p["character"][..., 1] = 40
    p["jumps_left"][..., 1] = 9
    p["hitstun"][..., 1] = 1000.0
    return p


def test_shapes_and_dtypes():
    f = codec.encode_state(players((4, 5)), 2)
    assert f["numeric"].shape == (4, 5, 2 * codec.NUMERIC_PER_PLAYER) and f["numeric"].dtype == np.float32
    for key in ("action", "character", "jumps"):
        assert f[key].shape == (4, 5, 2) and f[key].dtype == np.int64
    assert f["stage"].shape == (4, 5) and (f["stage"] == 2).all()


def test_agent_first_and_scaling():
    num = codec.encode_state(players(), np.array([0, 1, 2]))["numeric"]
    n = codec.NUMERIC_PER_PLAYER
    assert num[0, 0] == -3.0 and num[0, n] == 3.0  # x * 0.05, agent block first
    assert num[0, n + 2] == 0.5  # percent * 0.01
    assert num[0, 3] == 1.0  # stocks / 4
    assert num[0, 4] == 1.0 and num[0, n + 4] == -1.0  # facing
    assert num[0, n + 8] == 4.0  # hitstun / 30, capped at 4


def test_categories_are_clipped_to_their_tables():
    f = codec.encode_state(players(), 0)
    assert f["action"][0, 1] == codec.NUM_ACTIONS - 1
    assert f["character"][0, 1] == codec.NUM_CHARACTERS - 1
    assert f["jumps"][0, 1] == codec.NUM_JUMPS - 1
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_codec_features.py -v`
Expected: FAIL with `AttributeError: module 'smash_rl.codec' has no attribute 'encode_state'`.

- [ ] **Step 2: Add `encode_state` to `smash_rl/codec.py`**

Append to `smash_rl/codec.py` (and add `from smash_rl.state import NUM_STAGES` to its imports):

```python
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
    ], axis=-1).astype(np.float32)
    return {
        "numeric": numeric.reshape(numeric.shape[:-2] + (2 * NUMERIC_PER_PLAYER,)),
        "action": np.minimum(p["action"], NUM_ACTIONS - 1).astype(np.int64),
        "character": np.minimum(p["character"], NUM_CHARACTERS - 1).astype(np.int64),
        "jumps": np.minimum(p["jumps_left"], NUM_JUMPS - 1).astype(np.int64),
        "stage": np.broadcast_to(np.minimum(np.asarray(stage, dtype=np.int64), NUM_STAGES - 1),
                                 players.shape[:-1]).copy(),
    }
```

- [ ] **Step 3: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all pass.

---

### Task 7: Policy network (`smash_rl/model.py`)

**Files:**
- Create: `smash_rl/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `codec.ACTION_SIZES`, `NUM_ACTIONS`, `NUM_CHARACTERS`, `NUM_JUMPS`, `NUMERIC_PER_PLAYER`, `state.NUM_STAGES`.
- Produces: `class Policy(width=512, layers=3, ffw_mult=2, head_hidden=128)` with `initial_state(batch, device) -> list[(h, c)]`, `forward(inputs: dict of (B, T, ...) tensors with keys numeric action character jumps stage prev_action, state, reset: bool (B,)) -> (out (B, T, width), new_state)`, `loss(out, targets (B, T, 4)) -> (total, per_head list[4], accuracy list[4])`, `sample(out (B, width), temperature=1.0) -> (B, 4) long`, `checkpoint() -> dict`, `Policy.from_checkpoint(dict)`, `load_policy(path, device) -> Policy`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import torch

from smash_rl import codec
from smash_rl.model import Policy


def random_inputs(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    ints = lambda high, *shape: torch.randint(0, high, shape, generator=g)
    return {
        "numeric": torch.randn(B, T, 2 * codec.NUMERIC_PER_PLAYER, generator=g),
        "action": ints(codec.NUM_ACTIONS, B, T, 2), "character": ints(codec.NUM_CHARACTERS, B, T, 2),
        "jumps": ints(codec.NUM_JUMPS, B, T, 2), "stage": ints(7, B, T),
        "prev_action": torch.stack([ints(n, B, T) for n in codec.ACTION_SIZES], -1),
    }


def test_default_size_is_about_ten_million_parameters():
    n = sum(p.numel() for p in Policy().parameters())
    assert 8e6 < n < 12e6


def test_sequence_equals_frame_by_frame():
    torch.manual_seed(0)
    model = Policy(width=32, layers=2, head_hidden=16).eval()
    x = random_inputs(3, 6)
    reset = torch.zeros(3, dtype=torch.bool)
    full, _ = model(x, model.initial_state(3, "cpu"), reset)
    state, steps = model.initial_state(3, "cpu"), []
    for t in range(6):
        out, state = model({k: v[:, t:t + 1] for k, v in x.items()}, state, reset)
        steps.append(out)
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-5)


def test_reset_rows_start_from_a_fresh_state():
    torch.manual_seed(0)
    model = Policy(width=32, layers=2, head_hidden=16).eval()
    x = random_inputs(2, 4)
    _, state = model(x, model.initial_state(2, "cpu"), torch.zeros(2, dtype=torch.bool))
    after_reset, _ = model(x, state, torch.tensor([True, False]))
    fresh, _ = model(x, model.initial_state(2, "cpu"), torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(after_reset[0], fresh[0], atol=1e-6)
    assert not torch.allclose(after_reset[1], fresh[1], atol=1e-6)


def test_teacher_forced_loss_goes_down():
    torch.manual_seed(0)
    model = Policy(width=32, layers=1, head_hidden=16)
    x = random_inputs(4, 8)
    targets = torch.stack([torch.randint(0, n, (4, 8)) for n in codec.ACTION_SIZES], -1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        out, _ = model(x, model.initial_state(4, "cpu"), torch.zeros(4, dtype=torch.bool))
        loss, heads, accs = model.loss(out, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert len(heads) == 4 and len(accs) == 4
    assert losses[-1] < 0.5 * losses[0]


def test_samples_are_valid_classes():
    model = Policy(width=32, layers=1, head_hidden=16)
    out, _ = model(random_inputs(5, 1), model.initial_state(5, "cpu"), torch.zeros(5, dtype=torch.bool))
    a = model.sample(out[:, 0])
    assert a.shape == (5, 4)
    for k, n in enumerate(codec.ACTION_SIZES):
        assert ((a[:, k] >= 0) & (a[:, k] < n)).all()


def test_checkpoint_round_trip():
    model = Policy(width=32, layers=1, head_hidden=16)
    copy = Policy.from_checkpoint(model.checkpoint())
    for a, b in zip(model.state_dict().values(), copy.state_dict().values()):
        assert torch.equal(a, b)
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smash_rl.model'`.

- [ ] **Step 2: Implement `smash_rl/model.py`**

```python
"""Recurrent Melee policy: game state in, one controller (four classes) out per frame.

slippi-ai's "tx_like" design: an encoder, then blocks that alternate a residual LSTM
and a residual feed-forward layer, then an autoregressive controller head that picks
buttons, main stick x, main stick y and C-stick in turn, each seeing the earlier
choices. The LSTM state is carried through a whole game.
"""

import torch
import torch.nn.functional as F
from torch import nn

from smash_rl.codec import ACTION_SIZES, NUM_ACTIONS, NUM_CHARACTERS, NUM_JUMPS, NUMERIC_PER_PLAYER
from smash_rl.state import NUM_STAGES

CONTROLLER_EMB = (16, 8, 8, 4)  # embedding sizes of the four controller classes
ACTION_EMB, CHARACTER_EMB, JUMPS_EMB, STAGE_EMB = 32, 16, 4, 8


class LSTMBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lstm = nn.LSTM(width, width, batch_first=True)

    def forward(self, x, state):
        y, state = self.lstm(self.norm(x), state)
        return x + y, state


class FFWBlock(nn.Module):
    def __init__(self, width: int, mult: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * mult)
        self.down = nn.Linear(width * mult, width)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.norm(x))))


class Policy(nn.Module):
    def __init__(self, width: int = 512, layers: int = 3, ffw_mult: int = 2, head_hidden: int = 128):
        super().__init__()
        self.config = dict(width=width, layers=layers, ffw_mult=ffw_mult, head_hidden=head_hidden)
        self.action_emb = nn.Embedding(NUM_ACTIONS, ACTION_EMB)
        self.character_emb = nn.Embedding(NUM_CHARACTERS, CHARACTER_EMB)
        self.jumps_emb = nn.Embedding(NUM_JUMPS, JUMPS_EMB)
        self.stage_emb = nn.Embedding(NUM_STAGES, STAGE_EMB)
        self.prev_embs = nn.ModuleList(nn.Embedding(n, d) for n, d in zip(ACTION_SIZES, CONTROLLER_EMB))
        in_size = (2 * NUMERIC_PER_PLAYER + 2 * (ACTION_EMB + CHARACTER_EMB + JUMPS_EMB)
                   + STAGE_EMB + sum(CONTROLLER_EMB))
        self.encoder = nn.Linear(in_size, width)
        self.lstms = nn.ModuleList(LSTMBlock(width) for _ in range(layers))
        self.ffws = nn.ModuleList(FFWBlock(width, ffw_mult) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.head_embs = nn.ModuleList(nn.Embedding(n, d) for n, d in zip(ACTION_SIZES[:3], CONTROLLER_EMB[:3]))
        context = [0, CONTROLLER_EMB[0], sum(CONTROLLER_EMB[:2]), sum(CONTROLLER_EMB[:3])]
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(width + context[k], head_hidden), nn.GELU(), nn.Linear(head_hidden, n))
            for k, n in enumerate(ACTION_SIZES))

    def initial_state(self, batch: int, device) -> list:
        width = self.config["width"]
        return [(torch.zeros(1, batch, width, device=device), torch.zeros(1, batch, width, device=device))
                for _ in self.lstms]

    def forward(self, inputs: dict, state: list, reset: torch.Tensor):
        """inputs: (B, T, ...) tensors; reset: (B,) rows whose state starts fresh."""
        x = torch.cat([
            inputs["numeric"],
            self.action_emb(inputs["action"]).flatten(-2),
            self.character_emb(inputs["character"]).flatten(-2),
            self.jumps_emb(inputs["jumps"]).flatten(-2),
            self.stage_emb(inputs["stage"]),
            *(emb(inputs["prev_action"][..., k]) for k, emb in enumerate(self.prev_embs)),
        ], dim=-1)
        x = self.encoder(x)
        keep = (~reset)[None, :, None]
        new_state = []
        for lstm, ffw, (h, c) in zip(self.lstms, self.ffws, state):
            x, (h, c) = lstm(x, (h * keep.to(h.dtype), c * keep.to(c.dtype)))
            x = ffw(x)
            new_state.append((h, c))
        return self.out_norm(x), new_state

    def _logits(self, out, actions):
        """Logits of the four heads, each conditioned on `actions`' earlier classes."""
        logits, context = [], []
        for k, head in enumerate(self.heads):
            logits.append(head(torch.cat([out, *context], dim=-1)))
            if k < 3:
                context.append(self.head_embs[k](actions[..., k]))
        return logits

    def loss(self, out, targets):
        """Teacher-forced cross-entropy: (total, per-head losses, per-head accuracies)."""
        logits = self._logits(out, targets)
        losses = [F.cross_entropy(l.reshape(-1, l.shape[-1]).float(), targets[..., k].reshape(-1))
                  for k, l in enumerate(logits)]
        accuracies = [(l.argmax(-1) == targets[..., k]).float().mean() for k, l in enumerate(logits)]
        return sum(losses), losses, accuracies

    @torch.no_grad()
    def sample(self, out, temperature: float = 1.0):
        """(B, 4) controller classes for core outputs (B, width)."""
        actions, context = [], []
        for k, head in enumerate(self.heads):
            logits = head(torch.cat([out, *context], dim=-1)).float() / temperature
            a = torch.distributions.Categorical(logits=logits).sample()
            actions.append(a)
            if k < 3:
                context.append(self.head_embs[k](a))
        return torch.stack(actions, dim=-1)

    def checkpoint(self) -> dict:
        return {"config": dict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "Policy":
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


def load_policy(path: str, device) -> Policy:
    """A Policy from a train_bc.py checkpoint (or a bare Policy.checkpoint())."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return Policy.from_checkpoint(checkpoint.get("model", checkpoint)).to(device)
```

- [ ] **Step 3: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 6 passed.

---

### Task 8: Training batches (`smash_rl/data.py`)

**Files:**
- Create: `smash_rl/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `replays.read_index`, `codec.encode_state`, `codec.controller_classes`, `codec.NEUTRAL`.
- Produces: `load_index(data_dir) -> list[dict]`, `trajectory_list(index, split, data_dir) -> list[(path, slot)]`, `load_trajectory(path, slot) -> (players (T, 2) agent first, stage int)`, `class UnrollBatcher(trajectories, batch_size, unroll, delay, seed, loader=load_trajectory)` with `next_batch() -> dict` (numpy: `numeric (B, U, 32)`, `action character jumps (B, U, 2)`, `stage (B, U)`, `prev_action target (B, U, 4)`, `reset (B,) bool`), `class Prefetcher(batcher, depth=4)` with `next()`, `to_torch(batch, device) -> dict of tensors`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
import json

import numpy as np
import pytest

from smash_rl import codec
from smash_rl.data import UnrollBatcher, load_index, load_trajectory, trajectory_list
from smash_rl.state import PLAYER_DTYPE


def make_game(path, T, fox_slots=(0,)):
    p = np.zeros((T, 2), dtype=PLAYER_DTYPE)
    # Encode the frame number in the agent's stick x class: class(t) = t % 17.
    for slot in (0, 1):
        p[:, slot]["main_x"] = (np.arange(T) % 17) * 0.125 - 1.0
        p[:, slot]["x"] = slot  # tells the players apart
    np.savez_compressed(path, players=p, stage=np.int8(1), first_frame=np.int32(-123))


def make_dataset(tmp_path, lengths, ditto_first=False):
    lines = []
    for i, T in enumerate(lengths):
        rel = f"parsed/a/g{i}.npz"
        (tmp_path / "parsed" / "a").mkdir(parents=True, exist_ok=True)
        slots = [0, 1] if (ditto_first and i == 0) else [0]
        make_game(str(tmp_path / rel), T)
        lines.append(json.dumps({"game_id": f"g{i}", "path": rel, "slots": slots, "split": "train", "frames": T}))
    (tmp_path / "index.jsonl").write_text("\n".join(lines) + "\n")
    return trajectory_list(load_index(str(tmp_path)), "train", str(tmp_path))


def test_ditto_gives_two_trajectories_and_slot_one_swaps_players(tmp_path):
    trajs = make_dataset(tmp_path, [300, 300], ditto_first=True)
    assert len(trajs) == 3
    players, stage = load_trajectory(*[t for t in trajs if t[1] == 1][0])
    assert stage == 1 and (players[:, 0]["x"] == 1).all() and (players[:, 1]["x"] == 0).all()


def test_labels_are_delay_frames_ahead_and_prev_is_one_before(tmp_path):
    trajs = make_dataset(tmp_path, [400])
    b = UnrollBatcher(trajs, batch_size=2, unroll=20, delay=5, seed=0)
    first = b.next_batch()
    assert first["reset"].all()
    t = np.arange(20)
    assert (first["target"][0, :, 1] == (t + 5) % 17).all()
    assert first["prev_action"][0, 0].tolist() == codec.NEUTRAL.tolist()  # trajectory start
    assert (first["prev_action"][0, 1:, 1] == (t[1:] + 4) % 17).all()
    second = b.next_batch()
    assert not second["reset"].any()
    assert (second["target"][0, :, 1] == (t + 25) % 17).all()
    assert (second["prev_action"][0, :, 1] == (t + 24) % 17).all()


def test_batch_shapes(tmp_path):
    b = UnrollBatcher(make_dataset(tmp_path, [400, 500]), batch_size=3, unroll=10, delay=2, seed=0)
    batch = b.next_batch()
    assert batch["numeric"].shape == (3, 10, 2 * codec.NUMERIC_PER_PLAYER)
    assert batch["action"].shape == (3, 10, 2) and batch["stage"].shape == (3, 10)
    assert batch["target"].shape == (3, 10, 4) and batch["reset"].shape == (3,)


def test_short_trajectories_are_never_used(tmp_path):
    trajs = make_dataset(tmp_path, [24, 400])  # 24 < unroll + delay
    b = UnrollBatcher(trajs, batch_size=4, unroll=20, delay=5, seed=0)
    for _ in range(50):
        b.next_batch()
    assert all(len(r[1]) == 400 for r in b.readers)


def test_only_short_trajectories_raise(tmp_path):
    b = UnrollBatcher(make_dataset(tmp_path, [10, 12]), batch_size=1, unroll=20, delay=5, seed=0)
    with pytest.raises(RuntimeError, match="long enough"):
        b.next_batch()
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smash_rl.data'`.

- [ ] **Step 2: Implement `smash_rl/data.py`**

```python
"""Imitation-learning batches: stateful unrolls through whole games.

Each of the batch's rows walks through one trajectory (one player of one game) in
consecutive chunks, so the model's LSTM state can be carried from one batch to the
next exactly as when it plays a live game. A row that reaches the end of its
trajectory starts a random new one and is flagged `reset`.
"""

import os
import queue
import threading

import numpy as np
import torch

from smash_rl.codec import NEUTRAL, controller_classes, encode_state
from smash_rl.replays import read_index


def load_index(data_dir: str) -> list:
    return read_index(os.path.join(data_dir, "index.jsonl"))


def trajectory_list(index: list, split: str, data_dir: str) -> list:
    """(path, slot) for every kept player of every game in a split (a ditto gives two)."""
    return [(os.path.join(data_dir, e["path"]), slot)
            for e in index if e.get("split") == split for slot in e["slots"]]


def load_trajectory(path: str, slot: int):
    """(raw frames (T, 2) with the trajectory's player first, stage index)."""
    with np.load(path) as data:
        players = data["players"]
        stage = int(data["stage"])
    if slot == 1:
        players = players[:, ::-1]
    return np.ascontiguousarray(players), stage


class UnrollBatcher:
    """Batches of `unroll` frames: inputs at frames t, labels at t + delay."""

    def __init__(self, trajectories, batch_size: int, unroll: int, delay: int, seed: int, loader=load_trajectory):
        self.trajectories = list(trajectories)
        self.batch_size, self.unroll, self.delay = batch_size, unroll, delay
        self.rng = np.random.default_rng(seed)
        self.loader = loader
        self.readers = [None] * batch_size  # [features, classes, position]

    def _new_reader(self, i: int):
        for _ in range(1000):
            path, slot = self.trajectories[self.rng.integers(len(self.trajectories))]
            players, stage = self.loader(path, slot)
            if len(players) >= self.unroll + self.delay:
                self.readers[i] = [encode_state(players, stage), controller_classes(players[:, 0]), 0]
                return
        raise RuntimeError("no trajectory is long enough for unroll + delay frames")

    def next_batch(self) -> dict:
        U, d = self.unroll, self.delay
        reset = np.zeros(self.batch_size, dtype=bool)
        rows = []
        for i in range(self.batch_size):
            reader = self.readers[i]
            if reader is None or reader[2] + U + d > len(reader[1]):
                self._new_reader(i)
                reset[i] = True
                reader = self.readers[i]
            features, classes, p = reader
            prev = classes[p + d - 1:p + U + d - 1].copy()
            if p == 0:
                prev[0] = NEUTRAL  # live, the first frame has no previous decision either
            rows.append(({k: v[p:p + U] for k, v in features.items()}, prev, classes[p + d:p + U + d]))
            reader[2] = p + U
        batch = {k: np.stack([r[0][k] for r in rows]) for k in rows[0][0]}
        batch["prev_action"] = np.stack([r[1] for r in rows])
        batch["target"] = np.stack([r[2] for r in rows])
        batch["reset"] = reset
        return batch


class Prefetcher:
    """Builds batches in a background thread so the GPU doesn't wait for loading."""

    def __init__(self, batcher: UnrollBatcher, depth: int = 4):
        self.queue = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, args=(batcher,), daemon=True)
        self.thread.start()

    def _run(self, batcher):
        try:
            while True:
                self.queue.put(batcher.next_batch())
        except BaseException as e:  # surface loader errors in the training loop
            self.queue.put(e)

    def next(self) -> dict:
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item


def to_torch(batch: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in batch.items()}
```

- [ ] **Step 3: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all pass.

---

### Task 9: Imitation training (`train_bc.py`)

**Files:**
- Create: `train_bc.py`
- Modify: `README.md`, `AGENTS.md`

**Interfaces:**
- Consumes: `data.load_index`, `trajectory_list`, `UnrollBatcher`, `Prefetcher`, `to_torch`; `model.Policy`.
- Produces: run directory with `last.pt` and `best.pt` (`{"model": Policy.checkpoint(), "optimizer", "scaler", "step", "hours", "best_val", "args"}`) and TensorBoard logs (`train/loss`, `train/<head>_loss`, `train/<head>_acc`, `val/loss`, `val/<head>_acc`).

Mixed precision uses fp16 autocast with a GradScaler (cuDNN LSTMs support fp16 on every CUDA GPU; bf16 support for cuDNN RNNs is not guaranteed). This replaces the spec's "bf16".

- [ ] **Step 1: Implement `train_bc.py`**

```python
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
```

- [ ] **Step 2: Smoke run**

Run (data from Task 5's background run is enough once a few thousand games are indexed):
`.venv\Scripts\python.exe train_bc.py --run-dir runs\bc_smoke --max-hours 0.05 --batch-size 64 --eval-minutes 1 --val-batches 5 --log-every 20`
Expected: `~9.9M parameters`, training lines with the loss going down over ~3 minutes, at least two `val loss` lines, and `runs\bc_smoke\last.pt` plus `best.pt` present.

- [ ] **Step 3: Check resume**

Run: `.venv\Scripts\python.exe train_bc.py --run-dir runs\bc_smoke --resume --max-hours 0.02 --batch-size 64 --eval-minutes 1 --val-batches 5 --log-every 20`
Expected: `resumed at step <N>` and step numbers continuing from N.

- [ ] **Step 4: Check the full batch fits the GPU**

Run: `.venv\Scripts\python.exe train_bc.py --run-dir runs\bc_size --max-hours 0.03 --eval-minutes 100 --log-every 20`
Expected: no CUDA out-of-memory error; note the steps per second (step count / ~108 s). If it runs out of memory, retry with `--batch-size 384`, then `256`, and use the largest that fits for the overnight run (write it into README). Delete `runs\bc_size` afterwards.

- [ ] **Step 5: Document it**

README: an "Imitation training" section with the commands above, all flags, what `last.pt`/`best.pt` are, and the TensorBoard tags. AGENTS.md map: add `smash_rl/model.py`, `smash_rl/data.py`, `smash_rl/replays.py`, `prepare_data.py`, `train_bc.py`.

---

### Task 10: Playing the bot (`smash_rl/agent.py`, `evaluate.py`)

**Files:**
- Create: `smash_rl/agent.py`, `evaluate.py`, `tests/test_agent.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `model.load_policy`, `Policy.initial_state/forward/sample`, `codec.encode_state`, `codec.NEUTRAL`, `vec_env.make_vec_env`.
- Produces: `class Agent(model, n, device, temperature=1.0)` with `reset(i)` and `act(players (n, 2) PLAYER_DTYPE, stage (n,)) -> np.ndarray (n, 4)`; `evaluate.py` CLI.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent.py`:

```python
import numpy as np
import torch

from smash_rl import codec
from smash_rl.agent import Agent
from smash_rl.model import Policy
from smash_rl.state import PLAYER_DTYPE


def test_agent_acts_for_every_game_and_resets_one():
    torch.manual_seed(0)
    agent = Agent(Policy(width=32, layers=1, head_hidden=16), n=3, device="cpu")
    players = np.zeros((3, 2), dtype=PLAYER_DTYPE)
    actions = agent.act(players, np.zeros(3, dtype=np.int64))
    assert actions.shape == (3, 4)
    for k, size in enumerate(codec.ACTION_SIZES):
        assert ((actions[:, k] >= 0) & (actions[:, k] < size)).all()
    assert (agent.prev == actions).all()
    agent.reset(1)
    assert agent.prev[1].tolist() == codec.NEUTRAL.tolist()
    assert all(float(h[:, 1].abs().sum()) == 0 for h, _ in agent.state)
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smash_rl.agent'`.

- [ ] **Step 2: Implement `smash_rl/agent.py`**

```python
"""Runs a Policy live: one LSTM state and previous decision per game, one batched forward per frame."""

import numpy as np
import torch

from smash_rl.codec import NEUTRAL, encode_state
from smash_rl.data import to_torch


class Agent:
    def __init__(self, model, n: int, device, temperature: float = 1.0):
        self.model = model.to(device).eval()
        self.n, self.device, self.temperature = n, device, temperature
        self.state = model.initial_state(n, device)
        self.prev = np.tile(NEUTRAL, (n, 1))

    def reset(self, i: int):
        """Game i ended: its next frame starts a new game with a fresh state."""
        for h, c in self.state:
            h[:, i] = 0
            c[:, i] = 0
        self.prev[i] = NEUTRAL

    @torch.no_grad()
    def act(self, players, stage) -> np.ndarray:
        """Controller classes (n, 4) for raw frames (n, 2) [agent, opponent] and stage indices (n,)."""
        inputs = encode_state(players[:, None], np.asarray(stage)[:, None])
        inputs["prev_action"] = self.prev[:, None]
        batch = to_torch(inputs, self.device)
        reset = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        out, self.state = self.model(batch, self.state, reset)
        self.prev = self.model.sample(out[:, 0], self.temperature).cpu().numpy()
        return self.prev.copy()
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py -v`
Expected: 1 passed.

- [ ] **Step 3: Implement `evaluate.py`**

```python
"""Play a trained bot against the CPU, report its results and save replays.

Usage:
    python evaluate.py runs/bc_fox/best.pt --games 20 --n-envs 4            # fast, numbers
    python evaluate.py runs/bc_fox/best.pt --games 2 --n-envs 1 --watch     # normal speed, to watch
    python evaluate.py runs/bc_fox/best.pt --stage BATTLEFIELD --cpu-level 9 --replays runs/bc_fox/replays
"""

import argparse
import os

import numpy as np
import torch

from smash_rl.agent import Agent
from smash_rl.model import load_policy
from smash_rl.vec_env import make_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained bot against the CPU")
    p.add_argument("checkpoint", help="a .pt file from train_bc.py")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=4, help="games played at once")
    p.add_argument("--opponent", default="FOX", help="CPU character, or RANDOM")
    p.add_argument("--cpu-level", type=int, default=9)
    p.add_argument("--stage", default="FINAL_DESTINATION")
    p.add_argument("--delay", type=int, default=18, help="must match training")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (1 = as trained)")
    p.add_argument("--watch", action="store_true", help="run at normal speed (60 fps)")
    p.add_argument("--replays", default=None, help="directory for Slippi replays of the games")
    p.add_argument("--base-port", type=int, default=51600)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(load_policy(args.checkpoint, device), args.n_envs, device, args.temperature)
    env = make_vec_env(n_envs=args.n_envs, base_port=args.base_port, opponent=args.opponent,
                       cpu_level=args.cpu_level, stages=(args.stage,), delay=args.delay,
                       emulation_speed=1.0 if args.watch else 0.0,
                       save_replays=os.path.abspath(args.replays) if args.replays else None)
    results = []
    try:
        obs, _ = env.reset()
        while len(results) < args.games:
            obs, _, dones, infos = env.step(agent.act(obs["players"], obs["stage"]))
            for i in np.flatnonzero(dones):
                info = infos[i]
                results.append(info)
                agent.reset(i)
                minutes = max(info["frames"], 1) / 3600
                print(f"game {len(results)}: {'WON ' if info['won'] else 'lost'} vs {info['opponent']} | "
                      f"stocks {info['stock_diff']:+d} | kills {info['kills']} deaths {info['deaths']} | "
                      f"damage/min dealt {info['damage_dealt'] / minutes:.0f} taken {info['damage_taken'] / minutes:.0f}",
                      flush=True)
    finally:
        env.close()
    results = results[:args.games]
    print(f"\nwin rate {np.mean([r['won'] for r in results]):.0%} over {len(results)} games vs CPU level "
          f"{args.cpu_level} {args.opponent} on {args.stage} | mean stock diff "
          f"{np.mean([r['stock_diff'] for r in results]):+.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Live smoke test**

Run: `.venv\Scripts\python.exe evaluate.py runs\bc_smoke\best.pt --games 1 --n-envs 1 --cpu-level 1 --replays runs\bc_smoke\replays`
Expected: one Dolphin window; after the game, a `game 1: ...` line and the summary line; a `.slp` file under `runs\bc_smoke\replays`; no Dolphin left running.

- [ ] **Step 5: Document it**

README: an "Evaluate and watch" section with the commands and flags above, and how to open the saved `.slp` replays (Slippi Launcher's replay browser, or Dolphin's "Play replay" with the same ISO).

---

### Task 11: Overnight imitation training (operational)

**Files:** none (runs the pipeline).

- [ ] **Step 1: Confirm the data run finished**

Run: `.venv\Scripts\python.exe -c "from smash_rl.data import load_index; i = load_index(r'D:\smash_data'); print(len(i), sum(e['split'] == 'val' for e in i))"`
Expected: roughly 60-80k games in total (after filtering), about 2% of them val.

- [ ] **Step 2: Night 1**

Run: `.venv\Scripts\python.exe train_bc.py --run-dir runs\bc_fox --max-hours 10` (plus `--batch-size` from Task 9 Step 4 if it had to be lowered).
Expected next morning: `stopped after 10.00 h`; in TensorBoard (`.venv\Scripts\tensorboard.exe --logdir runs`), `val/loss` going down and per-head val accuracies rising (buttons and C-stick typically well above stick x/y).

- [ ] **Step 3: Night 2 if still improving**

If `val/loss` improved over the last 2 hours of night 1, run: `.venv\Scripts\python.exe train_bc.py --run-dir runs\bc_fox --resume --max-hours 10`. Otherwise go to Task 12 with `best.pt`.

---

### Task 12: Phase 1 evaluation and sign-off (operational)

**Files:**
- Modify: `README.md` (a "Results" section)

- [ ] **Step 1: Numbers against the level 9 CPU**

Run: `.venv\Scripts\python.exe evaluate.py runs\bc_fox\best.pt --games 20 --n-envs 4 --replays runs\bc_fox\eval_cpu9`
Expected: 20 games complete without any env getting stuck; record win rate and mean stock difference (target ≥ 70% wins).

- [ ] **Step 2: Watch it**

Run: `.venv\Scripts\python.exe evaluate.py runs\bc_fox\best.pt --games 2 --n-envs 1 --watch`
Check by eye, together with the user: short hops, L-cancels, wavedashes or waveshines, and combos of more than one hit. Also open two replays from `runs\bc_fox\eval_cpu9` in Slippi.

- [ ] **Step 3: Other stages sanity check**

Run: `.venv\Scripts\python.exe evaluate.py runs\bc_fox\best.pt --games 6 --n-envs 3 --stage BATTLEFIELD`
Expected: games complete; record the win rate (platform play is learned from the replays).

- [ ] **Step 4: Write the results down**

README "Results": training hours and steps, final val loss and per-head accuracies, the CPU 9 win rate and stock difference on FD and Battlefield, and what was seen in the replays. Phase 1 is done when the spec's four criteria hold; anything short of the 70% target goes into Phase 2's brainstorming as a known starting point.
