import pytest

from smash_rl import dolphin
from smash_rl.dolphin import Dolphin


class FakeConsole:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def step(self):
        self.calls += 1
        return self.results.pop(0) if self.results else None


def bare_dolphin(console, alive=True, human=False):
    d = Dolphin.__new__(Dolphin)  # no emulator: only next_state is exercised
    d.console = console
    d.human_opponent = human
    d.alive = lambda: alive
    return d


def test_next_state_keeps_asking_until_a_frame_arrives():
    console = FakeConsole([None, None, "frame"])
    assert bare_dolphin(console).next_state() == "frame"
    assert console.calls == 3  # each empty step re-sends the input (polling mode)


def test_next_state_gives_up_instead_of_hanging():
    with pytest.raises(RuntimeError, match="stopped sending"):
        bare_dolphin(FakeConsole([])).next_state(give_up=0.05)


def test_console_re_sends_the_input_when_dolphin_stalls(monkeypatch, tmp_path):
    # Dolphin occasionally needs an input on a frame that sends no game event (menus,
    # loading); a console that only waits would deadlock with it, so it polls instead.
    seen = {}

    def fake_console(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(dolphin, "_Console", fake_console)
    monkeypatch.setattr(dolphin, "_windows_dolphin_copy", lambda exe, port: (exe, None))
    with pytest.raises(RuntimeError, match="stop here"):
        Dolphin(51999).start()
    assert seen["polling_mode"] is True and seen["polling_timeout"] == dolphin.STALL_SECONDS


def test_a_closed_dolphin_is_reported_at_once():
    with pytest.raises(RuntimeError, match="closed"):
        bare_dolphin(FakeConsole([]), alive=False).next_state(give_up=1000)


def test_with_a_person_on_port_2_it_waits_without_limit():
    # A person may pause the game for as long as they like.
    console = FakeConsole([None] * 5 + ["frame"])
    d = bare_dolphin(console, human=True)
    assert d.next_state() == "frame"
