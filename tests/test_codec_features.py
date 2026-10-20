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


def test_non_finite_values_become_finite_inputs():
    p = players()
    p["x"][0, 0], p["speed_y"][0, 1] = np.nan, np.inf
    assert np.isfinite(codec.encode_state(p, 0)["numeric"]).all()
