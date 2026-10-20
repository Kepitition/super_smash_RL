import torch

from smash_rl import codec
from smash_rl.model import Policy


def random_inputs(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    ints = lambda high, *shape: torch.randint(0, high, shape, generator=g)
    return {
        "numeric": torch.randn(B, T, 2 * codec.NUMERIC_PER_PLAYER, generator=g),
        "action": ints(codec.NUM_ACTIONS, B, T, 2), "character": ints(codec.NUM_CHARACTERS, B, T, 2),
        "jumps": ints(codec.NUM_JUMPS, B, T, 2), "stage": ints(7, B, T),
        "prev_action": torch.stack([ints(n, B, T) for n in codec.ACTION_SIZES], -1),
    }


def test_default_size_is_about_ten_million_parameters():
    n = sum(p.numel() for p in Policy().parameters())
    assert 8e6 < n < 12e6


def test_sequence_equals_frame_by_frame():
    torch.manual_seed(0)
    model = Policy(width=32, layers=2, head_hidden=16).eval()
    x = random_inputs(3, 6)
    reset = torch.zeros(3, dtype=torch.bool)
    full, _ = model(x, model.initial_state(3, "cpu"), reset)
    state, steps = model.initial_state(3, "cpu"), []
    for t in range(6):
        out, state = model({k: v[:, t:t + 1] for k, v in x.items()}, state, reset)
        steps.append(out)
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-5)


def test_reset_rows_start_from_a_fresh_state():
    torch.manual_seed(0)
    model = Policy(width=32, layers=2, head_hidden=16).eval()
    x = random_inputs(2, 4)
    _, state = model(x, model.initial_state(2, "cpu"), torch.zeros(2, dtype=torch.bool))
    after_reset, _ = model(x, state, torch.tensor([True, False]))
    fresh, _ = model(x, model.initial_state(2, "cpu"), torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(after_reset[0], fresh[0], atol=1e-6)
    assert not torch.allclose(after_reset[1], fresh[1], atol=1e-6)


def test_teacher_forced_loss_goes_down():
    torch.manual_seed(0)
    model = Policy(width=32, layers=1, head_hidden=16)
    x = random_inputs(4, 8)
    targets = torch.stack([torch.randint(0, n, (4, 8)) for n in codec.ACTION_SIZES], -1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        out, _ = model(x, model.initial_state(4, "cpu"), torch.zeros(4, dtype=torch.bool))
        loss, heads, accs = model.loss(out, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert len(heads) == 4 and len(accs) == 4
    assert losses[-1] < 0.5 * losses[0]


def test_samples_are_valid_classes():
    model = Policy(width=32, layers=1, head_hidden=16)
    out, _ = model(random_inputs(5, 1), model.initial_state(5, "cpu"), torch.zeros(5, dtype=torch.bool))
    a = model.sample(out[:, 0])
    assert a.shape == (5, 4)
    for k, n in enumerate(codec.ACTION_SIZES):
        assert ((a[:, k] >= 0) & (a[:, k] < n)).all()


def test_checkpoint_round_trip():
    model = Policy(width=32, layers=1, head_hidden=16)
    copy = Policy.from_checkpoint(model.checkpoint())
    for a, b in zip(model.state_dict().values(), copy.state_dict().values()):
        assert torch.equal(a, b)


def test_reset_clears_a_non_finite_state():
    torch.manual_seed(0)
    model = Policy(width=32, layers=1, head_hidden=16).eval()
    x = random_inputs(2, 3)
    state = [(torch.full_like(h, float("nan")), torch.full_like(c, float("nan"))) for h, c in model.initial_state(2, "cpu")]
    out, _ = model(x, state, torch.tensor([True, True]))
    assert torch.isfinite(out).all()
