"""Our Fox for any libmelee match loop, with the same interface as the Ganondorf bot.

    bot = FoxBot("models/fox_rl_3h.pt", port=2, opponent_port=1)
    while True:
        gs = console.step()
        if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            bot.act(gs, controller)   # every in-game frame, before the next console.step()
        else:
            bot.reset()               # between games

It decides every frame and, like a person, reacts `delay` (18) frames late: each decision
waits in a queue before it reaches the controller, exactly as in training. Needs PyTorch
and libmelee; runs on a GPU when there is one, else on the CPU (~5-10 ms per frame).
"""

import collections
import os

import numpy as np
import torch

from smash_rl.agent import Agent
from smash_rl.codec import NEUTRAL, apply_controller
from smash_rl.model import load_policy
from smash_rl.state import from_libmelee, stage_index

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "fox_rl_3h.pt")


class FoxBot:
    character = "FOX"

    def __init__(self, model_path: str = DEFAULT_MODEL, port: int = 1, opponent_port: int = 2, delay: int = 18,
                 device: str = None, temperature: float = 1.0):
        self.model_path, self.port, self.opponent_port = model_path, port, opponent_port
        self.delay, self.device, self.temperature = delay, device, temperature
        self._agent = None  # loaded on first use, so the bot can be copied to other processes first
        self._queue = self._neutral_queue()

    def _neutral_queue(self):
        return collections.deque(NEUTRAL.copy() for _ in range(self.delay - 1))

    def _load(self):
        if self._agent is None:
            device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self._agent = Agent(load_policy(self.model_path, device), 1, device, self.temperature)

    def reset(self):
        """Call between games."""
        self._load()
        self._agent.reset(0)
        self._queue = self._neutral_queue()

    def act(self, gs, controller):
        """Set `controller` for the next frame. Call every in-game frame, before console.step()."""
        if self.port not in gs.players or self.opponent_port not in gs.players:
            return
        self._load()
        players = from_libmelee(gs, ports=(self.port, self.opponent_port))  # always [me, opponent]
        decision = self._agent.act(players[None], np.array([stage_index(gs.stage)]))[0]
        self._queue.append(decision)
        apply_controller(controller, self._queue.popleft())
