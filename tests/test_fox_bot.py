import melee
import numpy as np
import torch

from smash_rl import codec
from smash_rl.fox_bot import FoxBot
from smash_rl.model import Policy


class Pad:
    """Records the main stick of each frame."""

    def __init__(self):
        self.sticks = []

    def press_button(self, button):
        pass

    def release_button(self, button):
        pass

    def press_shoulder(self, button, amount):
        pass

    def tilt_analog_unit(self, button, x, y):
        if button == melee.Button.BUTTON_MAIN:
            self.sticks.append((x, y))


def game_state():
    gs = melee.GameState()
    gs.stage = melee.Stage.FINAL_DESTINATION
    for port in (1, 2):
        p = gs.players[port] = melee.PlayerState()
        p.character, p.action, p.stock, p.jumps_left = melee.Character.FOX, melee.Action.STANDING, 4, 2
    return gs


def test_decisions_reach_the_controller_delay_frames_later(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "fox.pt"
    torch.save({"model": Policy(width=32, layers=1, head_hidden=16).checkpoint()}, path)
    bot = FoxBot(str(path), port=2, opponent_port=1, delay=5, device="cpu")
    bot.reset()
    pad, gs = Pad(), game_state()
    decisions = []
    for _ in range(8):
        bot.act(gs, pad)
        decisions.append(bot._agent.prev[0].copy())
    neutral = (codec.STICK_SEND[codec.NEUTRAL[1]], codec.STICK_SEND[codec.NEUTRAL[2]])
    assert pad.sticks[:4] == [neutral] * 4  # the queue holds delay - 1 frames
    for frame in range(4, 8):
        made = decisions[frame - 4]
        assert pad.sticks[frame] == (codec.STICK_SEND[made[1]], codec.STICK_SEND[made[2]])


def test_missing_players_are_ignored():
    bot = FoxBot("never-loaded.pt", port=3, opponent_port=1)
    pad = Pad()
    bot.act(game_state(), pad)
    assert pad.sticks == [] and bot._agent is None
