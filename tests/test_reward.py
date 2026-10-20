import melee
import numpy as np
import pytest

from smash_rl.reward import EDGE_CATCHING, Reward
from smash_rl.state import PLAYER_DTYPE


def frame(me_x=0.0, op_x=30.0, me_pct=0.0, op_pct=0.0, me_stocks=4, op_stocks=4, me_action=14, me_y=0.0):
    f = np.zeros(2, dtype=PLAYER_DTYPE)
    f[0] = 0
    f["x"] = (me_x, op_x)
    f["y"] = (me_y, 0.0)
    f["percent"] = (me_pct, op_pct)
    f["stocks"] = (me_stocks, op_stocks)
    f["action"] = (me_action, 14)
    return f


def only(**weights):
    base = dict(kill=0.0, damage=0.0, ledge_grab=0.0, approach=0.0, stall=0.0)
    base.update(weights)
    return Reward(**base)


def test_kills_and_deaths_are_zero_sum():
    r = only(kill=1.0)
    assert r(frame(), frame(op_stocks=3)) == 1.0
    assert r(frame(), frame(me_stocks=3)) == -1.0


def test_damage_counts_within_a_stock_only():
    r = only(damage=0.01)
    assert r(frame(op_pct=10), frame(op_pct=25)) == pytest.approx(0.15)
    assert r(frame(me_pct=10), frame(me_pct=30)) == pytest.approx(-0.2)
    # Percent goes back to 0 when a stock is lost: no negative damage.
    assert r(frame(op_pct=120), frame(op_pct=0, op_stocks=3)) == 0.0


def test_grabbing_the_ledge_costs_once():
    r = only(ledge_grab=0.02)
    assert r(frame(), frame(me_action=EDGE_CATCHING)) == pytest.approx(-0.02)
    assert r(frame(me_action=EDGE_CATCHING), frame(me_action=EDGE_CATCHING)) == 0.0


def test_moving_towards_the_opponent_earns_a_little():
    r = only(approach=0.001)
    assert r(frame(me_x=0), frame(me_x=2)) == pytest.approx(0.002)
    assert r(frame(me_x=0), frame(me_x=-2)) == pytest.approx(-0.002)
    assert r(frame(me_x=0), frame(me_x=50, me_stocks=3)) == 0.0  # respawn, not movement


def test_staying_offstage_too_long_is_penalised():
    r = only(stall=0.01, stall_frames=2)
    r.reset(melee.Stage.FINAL_DESTINATION)
    off = frame(me_x=100.0)
    assert [r(off, off) for _ in range(4)] == [0.0, 0.0, -0.01, -0.01]
    assert r(off, frame(me_x=50.0)) == 0.0  # back on stage
    assert r(frame(me_x=50.0), off) == 0.0
