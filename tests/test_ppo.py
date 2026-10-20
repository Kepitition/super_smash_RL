import copy

import numpy as np
import pytest
import torch

from smash_rl.agent import Agent
from smash_rl.codec import encode_state
from smash_rl.data import to_torch
from smash_rl.model import Policy
from smash_rl.ppo import Learner, Rollout, ValueNet, gae
from smash_rl.state import PLAYER_DTYPE


def tiny_policy(seed=0):
    torch.manual_seed(seed)
    return Policy(width=32, layers=2, head_hidden=16)


def random_frames(n, length, seed=0):
    rng = np.random.default_rng(seed)
    players = np.zeros((n, length, 2), dtype=PLAYER_DTYPE)
    for field in ("x", "y", "percent", "speed_air_x", "action_frame"):
        players[field] = rng.normal(0, 20, players.shape)
    players["action"] = rng.integers(0, 380, players.shape)
    players["stocks"] = 4
    players["character"] = 1
    return players


def test_gae_stops_at_the_end_of_a_game():
    rewards = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    values = np.zeros((1, 4), dtype=np.float32)
    assert gae(rewards, values, np.array([[0, 0, 0]]), 1.0, 1.0).tolist() == [[1.0, 1.0, 1.0]]
    assert gae(rewards, values, np.array([[0, 1, 0]]), 1.0, 1.0).tolist() == [[0.0, 0.0, 1.0]]


def test_gae_bootstraps_from_the_last_value():
    adv = gae(np.zeros((1, 2), dtype=np.float32), np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
              np.zeros((1, 2)), 0.5, 1.0)
    assert adv.tolist() == [[0.5, 1.0]]


def test_a_game_starting_mid_sequence_restarts_the_lstm():
    model = tiny_policy().eval()
    players = random_frames(2, 6)
    inputs = encode_state(players, np.zeros((2, 6), dtype=np.int64))
    inputs["prev_action"] = np.zeros((2, 6, 4), dtype=np.int64)
    batch = to_torch(inputs, "cpu")
    reset = torch.zeros(2, 6, dtype=torch.bool)
    reset[0, 3] = True  # game 0 starts a new game at frame 3
    with torch.no_grad():
        out, _ = model(batch, model.initial_state(2, "cpu"), reset)
        whole, _ = model(batch, model.initial_state(2, "cpu"), torch.zeros(2, dtype=torch.bool))
        fresh, _ = model({k: v[:, 3:] for k, v in batch.items()}, model.initial_state(2, "cpu"),
                         torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(out[0, 3:], fresh[0], atol=1e-5)
    assert torch.allclose(out[1], whole[1], atol=1e-5)
    assert torch.allclose(out[0, :3], whole[0, :3], atol=1e-5)


def collect(policy, n=3, length=8, chunk=4, game_end=(1, 5)):
    """A rollout acted by `policy` on random frames; game `game_end[0]` ends after step game_end[1]."""
    agent = Agent(policy, n, "cpu")
    frames = random_frames(n, length + 1, seed=1)
    stage = np.zeros(n, dtype=np.int64)
    rollout = Rollout(n, length, chunk)
    first = np.ones(n, dtype=bool)
    for t in range(length):
        if t % chunk == 0:
            rollout.chunk_states.append(agent.state_copy())
        rollout.add_frame(frames[:, t], stage, first, agent.prev)
        actions = agent.act(frames[:, t], stage)
        dones = np.zeros(n, dtype=bool)
        if t == game_end[1]:
            dones[game_end[0]] = True
        rewards = np.where(dones, 1.0, 0.01).astype(np.float32)
        rollout.add_step(actions, agent.logp, rewards, dones)
        first = dones.copy()
        for i in np.flatnonzero(dones):
            agent.reset(i)
    rollout.add_frame(frames[:, length], stage, first, agent.prev)
    return rollout


def test_first_update_starts_on_policy_and_at_the_teacher():
    policy = tiny_policy()
    teacher = copy.deepcopy(policy)
    learner = Learner(policy, teacher, ValueNet.from_policy(policy), 3, "cpu", epochs=1, minibatches=1, lr=1e-3)
    before = [p.clone() for p in policy.parameters()]
    stats = learner.update(collect(policy))
    # The minibatch replays acting exactly: ratio 1, nothing clipped, policy == teacher.
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-5)
    assert stats["clip_frac"] == 0.0
    assert stats["teacher_kl"] == pytest.approx(0.0, abs=1e-5)
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    assert all(not p.requires_grad for p in teacher.parameters())


def test_value_warmup_leaves_the_policy_alone():
    policy = tiny_policy()
    value = ValueNet.from_policy(policy)
    learner = Learner(policy, copy.deepcopy(policy), value, 3, "cpu", value_lr=1e-3)
    before_policy = [p.clone() for p in policy.parameters()]
    before_value = [p.clone() for p in value.parameters()]
    learner.update(collect(policy), train_policy=False)
    assert all(torch.equal(a, b) for a, b in zip(before_policy, policy.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_value, value.parameters()))


def test_kl_grows_as_the_policy_moves_away():
    policy = tiny_policy()
    teacher = copy.deepcopy(policy)
    learner = Learner(policy, teacher, ValueNet.from_policy(policy), 3, "cpu", lr=1e-2, kl_coef=0.0)
    learner.update(collect(policy))
    assert learner.update(collect(policy))["teacher_kl"] > 1e-4


def test_value_net_starts_at_zero_and_round_trips():
    value = ValueNet.from_policy(tiny_policy())
    batch = encode_state(random_frames(2, 3), np.zeros((2, 3), dtype=np.int64))
    batch["prev_action"] = np.zeros((2, 3, 4), dtype=np.int64)
    with torch.no_grad():
        v, _ = value(to_torch(batch, "cpu"), value.initial_state(2, "cpu"), torch.zeros(2, dtype=torch.bool))
    assert v.shape == (2, 3) and float(v.abs().max()) == 0.0
    clone = ValueNet.from_checkpoint(value.checkpoint())
    assert all(torch.equal(a, b) for a, b in zip(clone.parameters(), value.parameters()))


def test_rollout_length_must_fit_whole_chunks():
    with pytest.raises(ValueError):
        Rollout(2, 10, 4)
