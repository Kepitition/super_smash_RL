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
