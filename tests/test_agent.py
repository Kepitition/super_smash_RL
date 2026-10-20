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
