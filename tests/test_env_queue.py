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
