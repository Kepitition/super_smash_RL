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


def test_rollback_frames_keep_the_last_copy_of_each_frame():
    # Online replays re-send frames after a rollback; the last copy is the confirmed one.
    g = fake_game(T=7)
    g.frames.id = np.array([-123, -122, -121, -122, -121, -120, -119], dtype=np.int32)
    g.frames.ports[0].leader.post.position.x = np.arange(7, dtype=np.float32)
    players, _, frame_ids = from_peppi(g)
    assert frame_ids.tolist() == [-123, -122, -121, -120, -119]
    assert players[:, 0]["x"].tolist() == [0, 3, 4, 5, 6]


def test_replay_with_missing_frames_raises_missing_data():
    g = fake_game(T=5)
    g.frames.id = np.array([-123, -122, -120, -119, -118], dtype=np.int32)
    with pytest.raises(state.MissingData, match="skips frames"):
        from_peppi(g)
