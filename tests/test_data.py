import json

import numpy as np
import pytest

from smash_rl import codec
from smash_rl.data import UnrollBatcher, load_index, load_trajectory, trajectory_list
from smash_rl.state import PLAYER_DTYPE


def make_game(path, T, fox_slots=(0,)):
    p = np.zeros((T, 2), dtype=PLAYER_DTYPE)
    # Encode the frame number in the agent's stick x class: class(t) = t % 17.
    for slot in (0, 1):
        p[:, slot]["main_x"] = (np.arange(T) % 17) * 0.125 - 1.0
        p[:, slot]["x"] = slot  # tells the players apart
    np.savez_compressed(path, players=p, stage=np.int8(1), first_frame=np.int32(-123))


def make_dataset(tmp_path, lengths, ditto_first=False):
    lines = []
    for i, T in enumerate(lengths):
        rel = f"parsed/a/g{i}.npz"
        (tmp_path / "parsed" / "a").mkdir(parents=True, exist_ok=True)
        slots = [0, 1] if (ditto_first and i == 0) else [0]
        make_game(str(tmp_path / rel), T)
        lines.append(json.dumps({"game_id": f"g{i}", "path": rel, "slots": slots, "split": "train", "frames": T}))
    (tmp_path / "index.jsonl").write_text("\n".join(lines) + "\n")
    return trajectory_list(load_index(str(tmp_path)), "train", str(tmp_path))


def test_ditto_gives_two_trajectories_and_slot_one_swaps_players(tmp_path):
    trajs = make_dataset(tmp_path, [300, 300], ditto_first=True)
    assert len(trajs) == 3
    players, stage = load_trajectory(*[t for t in trajs if t[1] == 1][0])
    assert stage == 1 and (players[:, 0]["x"] == 1).all() and (players[:, 1]["x"] == 0).all()


def test_labels_are_delay_frames_ahead_and_prev_is_one_before(tmp_path):
    trajs = make_dataset(tmp_path, [400])
    b = UnrollBatcher(trajs, batch_size=2, unroll=20, delay=5, seed=0)
    first = b.next_batch()
    assert first["reset"].all()
    t = np.arange(20)
    assert (first["target"][0, :, 1] == (t + 5) % 17).all()
    assert first["prev_action"][0, 0].tolist() == codec.NEUTRAL.tolist()  # trajectory start
    assert (first["prev_action"][0, 1:, 1] == (t[1:] + 4) % 17).all()
    second = b.next_batch()
    assert not second["reset"].any()
    assert (second["target"][0, :, 1] == (t + 25) % 17).all()
    assert (second["prev_action"][0, :, 1] == (t + 24) % 17).all()


def test_batch_shapes(tmp_path):
    b = UnrollBatcher(make_dataset(tmp_path, [400, 500]), batch_size=3, unroll=10, delay=2, seed=0)
    batch = b.next_batch()
    assert batch["numeric"].shape == (3, 10, 2 * codec.NUMERIC_PER_PLAYER)
    assert batch["action"].shape == (3, 10, 2) and batch["stage"].shape == (3, 10)
    assert batch["target"].shape == (3, 10, 4) and batch["reset"].shape == (3,)


def test_short_trajectories_are_never_used(tmp_path):
    trajs = make_dataset(tmp_path, [24, 400])  # 24 < unroll + delay
    b = UnrollBatcher(trajs, batch_size=4, unroll=20, delay=5, seed=0)
    for _ in range(50):
        b.next_batch()
    assert all(len(r[1]) == 400 for r in b.readers)


def test_only_short_trajectories_raise(tmp_path):
    b = UnrollBatcher(make_dataset(tmp_path, [10, 12]), batch_size=1, unroll=20, delay=5, seed=0)
    with pytest.raises(RuntimeError, match="long enough"):
        b.next_batch()
