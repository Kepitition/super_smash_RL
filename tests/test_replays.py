import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from smash_rl import replays
from smash_rl.state import FOX


def game(players=2, human=True, stage=0x20, end=None, play_frames=2000, characters=(FOX, 9), last_stocks=(2, 0)):
    T = 123 + play_frames
    ports = []
    for c, last in zip(characters[:players], list(last_stocks) + [4] * players):
        stocks = np.full(T, 4, np.uint8)
        stocks[-1] = last
        ports.append(NS(leader=NS(post=NS(character=np.full(T, c, np.uint8), stocks=stocks))))
    return NS(
        start=NS(players=[NS(type=0 if human else 1) for _ in range(players)], stage=stage),
        end=None if end is None else NS(method=end),  # the ranked dumps have no end record
        frames=NS(id=np.arange(-123, play_frames, dtype=np.int32), ports=tuple(ports)),
    )


def test_accepts_a_normal_fox_game():
    assert replays.check_game(game(), FOX) == [0]


def test_a_game_that_ran_out_of_time_is_kept():
    assert replays.check_game(game(play_frames=replays.TIMEOUT_FRAMES + 1, last_stocks=(2, 1)), FOX) == [0]


def test_fox_ditto_gives_both_slots():
    assert replays.check_game(game(characters=(FOX, FOX)), FOX) == [0, 1]


@pytest.mark.parametrize("kwargs, reason", [
    (dict(players=3, characters=(FOX, 9, 9)), "not 2 players"),
    (dict(human=False), "not human"),
    (dict(stage=0x99), "stage"),
    (dict(end=7), "quit or disconnect"),
    (dict(last_stocks=(2, 1)), "quit or disconnect"),
    (dict(play_frames=1000), "short"),
    (dict(characters=(9, 9)), "character absent"),
])
def test_rejects(kwargs, reason):
    with pytest.raises(replays.Skip, match=reason):
        replays.check_game(game(**kwargs), FOX)


def test_split_is_deterministic_and_about_two_percent():
    splits = [replays.game_split(f"g{i}") for i in range(20000)]
    assert splits == [replays.game_split(f"g{i}") for i in range(20000)]
    assert 0.01 < splits.count("val") / len(splits) < 0.03


def test_corrupt_replay_is_skipped(tmp_path):
    out = tmp_path / "x.npz"
    assert replays.convert_safely(b"not a replay", "x", str(out), FOX) == {"skip": "parse error"}
    assert not out.exists() and not list(tmp_path.iterdir())


def test_replay_without_speeds_is_skipped(tmp_path, monkeypatch):
    g = game()
    for port in g.frames.ports:
        port.leader.post.velocities = None
        port.leader.pre = None
    monkeypatch.setattr(replays.peppi_py, "read_slippi", lambda path: g)
    assert replays.convert_safely(b"x", "g", str(tmp_path / "g.npz"), FOX) == {"skip": "missing data"}
    assert not list(tmp_path.iterdir())


def test_read_index_drops_a_half_written_last_line(tmp_path):
    path = tmp_path / "index.jsonl"
    path.write_text(json.dumps({"game_id": "a"}) + "\n" + '{"game_id": "b", "fra')
    assert [e["game_id"] for e in replays.read_index(str(path))] == ["a"]
    assert replays.read_index(str(tmp_path / "missing.jsonl")) == []


def test_short_is_counted_in_distinct_frames_not_rollback_rows():
    g = game(play_frames=1000)
    g.frames.id = np.repeat(g.frames.id, 2)  # every frame sent twice, as after rollbacks
    for port in g.frames.ports:
        port.leader.post.character = np.repeat(port.leader.post.character, 2)
        port.leader.post.stocks = np.repeat(port.leader.post.stocks, 2)
    with pytest.raises(replays.Skip, match="short"):
        replays.check_game(g, FOX)


def test_any_error_after_parsing_skips_the_replay(tmp_path, monkeypatch):
    g = game()
    g.frames.id = np.zeros(0, dtype=np.int32)  # a readable replay without frames
    monkeypatch.setattr(replays.peppi_py, "read_slippi", lambda path: g)
    result = replays.convert_safely(b"x", "g", str(tmp_path / "g.npz"), FOX)
    assert result["skip"].startswith("bad replay")


def test_rewrite_index_keeps_the_old_file_if_writing_fails(tmp_path, monkeypatch):
    path = tmp_path / "index.jsonl"
    original = json.dumps({"game_id": "a"}) + "\n"
    path.write_text(original)
    monkeypatch.setattr(replays.json, "dumps", lambda e: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        replays.rewrite_index(str(path), [{"game_id": "a"}, {"game_id": "b"}])
    assert path.read_text() == original
