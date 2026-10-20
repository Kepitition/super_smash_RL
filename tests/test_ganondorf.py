import copy
from types import SimpleNamespace as NS

import cloudpickle
import melee
import numpy as np
import pytest

from smash_rl.env import SmashEnv
from smash_rl.ganondorf import ACTIONS, MIRRORED_ACTION, GanondorfBot

B = melee.Button
FORWARD_A = ACTIONS.index(((1.0, 0.5), B.BUTTON_A))


def bot_that_always_picks(action: int) -> GanondorfBot:
    weights = {"w0": np.zeros((256, 847)), "b0": np.zeros(256), "w1": np.zeros((256, 256)), "b1": np.zeros(256),
               "w2": np.zeros((31, 256)), "b2": np.eye(31)[action]}
    return GanondorfBot(weights, {"character": "GANONDORF"})


def player(x: float, character=melee.Character.FOX):
    return NS(position=NS(x=x, y=0.0), percent=0.0, stock=4, facing=True, on_ground=True, off_stage=False,
              invulnerable=False, hitlag_left=0, hitstun_frames_left=0, jumps_left=2, shield_strength=60.0,
              speed_air_x_self=0.0, speed_ground_x_self=0.0, speed_y_self=0.0, speed_x_attack=0.0,
              speed_y_attack=0.0, action_frame=1, iasa=0, action=14, character=character)


class Pad:
    def __init__(self):
        self.calls = []

    def release_all(self):
        self.calls.append("release_all")

    def tilt_analog(self, button, x, y):
        self.calls.append(("stick", x, y))

    def press_button(self, button):
        self.calls.append(("press", button))

    def release_button(self, button):
        self.calls.append(("release", button))


def game(bot_x: float, opponent_x: float):
    return NS(frame=0, players={2: player(bot_x, melee.Character.GANONDORF), 1: player(opponent_x)})


def test_mirroring_flips_only_the_stick_x():
    for i, (stick, button) in enumerate(ACTIONS):
        stick_m, button_m = ACTIONS[MIRRORED_ACTION[i]]
        assert button_m == button and stick_m[1] == stick[1]
        assert stick_m[0] == pytest.approx(1.0 - stick[0])
        assert MIRRORED_ACTION[MIRRORED_ACTION[i]] == i


def test_decides_every_third_frame_and_taps_buttons():
    bot, pad = bot_that_always_picks(FORWARD_A), Pad()
    for _ in range(4):
        bot.act(game(0.0, 30.0), pad)
    assert pad.calls == ["release_all", ("stick", 1.0, 0.5), ("press", B.BUTTON_A),  # frame 1: decide
                         ("release", B.BUTTON_A),                                     # frame 3: let go
                         "release_all", ("stick", 1.0, 0.5), ("press", B.BUTTON_A)]   # frame 4: decide


def test_the_opponent_on_the_left_mirrors_the_stick():
    # It was trained with the opponent always on its right: "forward" means towards them.
    bot, pad = bot_that_always_picks(FORWARD_A), Pad()
    bot.act(game(0.0, -30.0), pad)
    assert ("stick", 0.0, 0.5) in pad.calls


def test_reset_starts_a_new_decision():
    bot, pad = bot_that_always_picks(FORWARD_A), Pad()
    bot.act(game(0.0, 30.0), pad)
    bot.reset()
    pad.calls.clear()
    bot.act(game(0.0, 30.0), pad)
    assert pad.calls[0] == "release_all"


def test_wrong_input_size_is_refused():
    weights = {"w0": np.zeros((256, 846)), "b0": np.zeros(256), "w1": np.zeros((256, 256)), "b1": np.zeros(256),
               "w2": np.zeros((31, 256)), "b2": np.zeros(31)}
    with pytest.raises(ValueError, match="inputs"):
        GanondorfBot(weights, {})


def test_bot_travels_to_workers_and_each_env_gets_its_own():
    bot = bot_that_always_picks(FORWARD_A)
    clone = cloudpickle.loads(cloudpickle.dumps(bot))
    assert np.array_equal(clone.weights["b2"], bot.weights["b2"])
    env = SmashEnv(opponent_bot=bot)
    assert env.opponent == melee.Character.GANONDORF
    assert env.opponent_bot is not bot
    assert env.dolphin.instant_restart  # a fixed opponent on one stage
    assert copy.deepcopy(bot).character == "GANONDORF"
