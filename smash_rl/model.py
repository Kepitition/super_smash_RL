"""Recurrent Melee policy: game state in, one controller (four classes) out per frame.

slippi-ai's "tx_like" design: an encoder, then blocks that alternate a residual LSTM
and a residual feed-forward layer, then an autoregressive controller head that picks
buttons, main stick x, main stick y and C-stick in turn, each seeing the earlier
choices. The LSTM state is carried through a whole game.
"""

import torch
import torch.nn.functional as F
from torch import nn

from smash_rl.codec import ACTION_SIZES, NUM_ACTIONS, NUM_CHARACTERS, NUM_JUMPS, NUMERIC_PER_PLAYER
from smash_rl.state import NUM_STAGES

CONTROLLER_EMB = (16, 8, 8, 4)  # embedding sizes of the four controller classes
ACTION_EMB, CHARACTER_EMB, JUMPS_EMB, STAGE_EMB = 32, 16, 4, 8


class LSTMBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lstm = nn.LSTM(width, width, batch_first=True)

    def forward(self, x, state):
        y, state = self.lstm(self.norm(x), state)
        return x + y, state


class FFWBlock(nn.Module):
    def __init__(self, width: int, mult: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * mult)
        self.down = nn.Linear(width * mult, width)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.norm(x))))


class Policy(nn.Module):
    def __init__(self, width: int = 512, layers: int = 3, ffw_mult: int = 2, head_hidden: int = 128):
        super().__init__()
        self.config = dict(width=width, layers=layers, ffw_mult=ffw_mult, head_hidden=head_hidden)
        self.action_emb = nn.Embedding(NUM_ACTIONS, ACTION_EMB)
        self.character_emb = nn.Embedding(NUM_CHARACTERS, CHARACTER_EMB)
        self.jumps_emb = nn.Embedding(NUM_JUMPS, JUMPS_EMB)
        self.stage_emb = nn.Embedding(NUM_STAGES, STAGE_EMB)
        self.prev_embs = nn.ModuleList(nn.Embedding(n, d) for n, d in zip(ACTION_SIZES, CONTROLLER_EMB))
        in_size = (2 * NUMERIC_PER_PLAYER + 2 * (ACTION_EMB + CHARACTER_EMB + JUMPS_EMB)
                   + STAGE_EMB + sum(CONTROLLER_EMB))
        self.encoder = nn.Linear(in_size, width)
        self.lstms = nn.ModuleList(LSTMBlock(width) for _ in range(layers))
        self.ffws = nn.ModuleList(FFWBlock(width, ffw_mult) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.head_embs = nn.ModuleList(nn.Embedding(n, d) for n, d in zip(ACTION_SIZES[:3], CONTROLLER_EMB[:3]))
        context = [0, CONTROLLER_EMB[0], sum(CONTROLLER_EMB[:2]), sum(CONTROLLER_EMB[:3])]
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(width + context[k], head_hidden), nn.GELU(), nn.Linear(head_hidden, n))
            for k, n in enumerate(ACTION_SIZES))

    def initial_state(self, batch: int, device) -> list:
        width = self.config["width"]
        return [(torch.zeros(1, batch, width, device=device), torch.zeros(1, batch, width, device=device))
                for _ in self.lstms]

    def forward(self, inputs: dict, state: list, reset: torch.Tensor):
        """inputs: (B, T, ...) tensors. reset: (B,) rows whose state starts fresh, or (B, T)
        rows whose game starts at frame t (RL rollouts, where games end mid-sequence)."""
        x = torch.cat([
            inputs["numeric"],
            self.action_emb(inputs["action"]).flatten(-2),
            self.character_emb(inputs["character"]).flatten(-2),
            self.jumps_emb(inputs["jumps"]).flatten(-2),
            self.stage_emb(inputs["stage"]),
            *(emb(inputs["prev_action"][..., k]) for k, emb in enumerate(self.prev_embs)),
        ], dim=-1)
        x = self.encoder(x)
        if reset.dim() == 1:
            spans = [(0, x.shape[1], reset)]
        else:
            # Games start rarely, so the LSTMs run in one piece between the frames where one does.
            cuts = (reset[:, 1:].any(0).nonzero().flatten() + 1).tolist()
            bounds = [0] + cuts + [x.shape[1]]
            spans = [(s, e, reset[:, s]) for s, e in zip(bounds[:-1], bounds[1:])]
        new_state = []
        for lstm, ffw, (h, c) in zip(self.lstms, self.ffws, state):
            ys = []
            for start, end, fresh in spans:
                keep = (~fresh)[None, :, None]
                # where(), not a multiply: NaN * 0 would keep a poisoned state alive across the reset.
                h, c = torch.where(keep, h, torch.zeros_like(h)), torch.where(keep, c, torch.zeros_like(c))
                y, (h, c) = lstm(x[:, start:end], (h, c))
                ys.append(y)
            x = ffw(ys[0] if len(ys) == 1 else torch.cat(ys, dim=1))
            new_state.append((h, c))
        return self.out_norm(x), new_state

    def _logits(self, out, actions):
        """Logits of the four heads, each conditioned on `actions`' earlier classes."""
        logits, context = [], []
        for k, head in enumerate(self.heads):
            logits.append(head(torch.cat([out, *context], dim=-1)))
            if k < 3:
                context.append(self.head_embs[k](actions[..., k]))
        return logits

    def loss(self, out, targets):
        """Teacher-forced cross-entropy: (total, per-head losses, per-head accuracies)."""
        logits = self._logits(out, targets)
        losses = [F.cross_entropy(l.reshape(-1, l.shape[-1]).float(), targets[..., k].reshape(-1))
                  for k, l in enumerate(logits)]
        accuracies = [(l.argmax(-1) == targets[..., k]).float().mean() for k, l in enumerate(logits)]
        return sum(losses), losses, accuracies

    def log_probs(self, out, actions) -> list:
        """Per-head log-probabilities (..., n_k), each conditioned on `actions`' earlier classes."""
        return [F.log_softmax(l.float(), dim=-1) for l in self._logits(out, actions)]

    @torch.no_grad()
    def sample_with_logp(self, out, temperature: float = 1.0):
        """(B, 4) controller classes for core outputs (B, width) and their log-probability (B,)."""
        actions, context, logp = [], [], 0.0
        for k, head in enumerate(self.heads):
            log_p = F.log_softmax(head(torch.cat([out, *context], dim=-1)).float() / temperature, dim=-1)
            # Gumbel-max sampling: unlike Categorical it needs no GPU sync (2x faster per frame).
            noise = -torch.log(-torch.log(torch.rand_like(log_p).clamp_(1e-20, 1.0)))
            a = (log_p + noise).argmax(-1)
            logp = logp + log_p.gather(-1, a[:, None])[:, 0]
            actions.append(a)
            if k < 3:
                context.append(self.head_embs[k](a))
        return torch.stack(actions, dim=-1), logp

    def sample(self, out, temperature: float = 1.0):
        """(B, 4) controller classes for core outputs (B, width)."""
        return self.sample_with_logp(out, temperature)[0]

    def checkpoint(self) -> dict:
        return {"config": dict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "Policy":
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


def load_policy(path: str, device) -> Policy:
    """A Policy from a train_bc.py checkpoint (or a bare Policy.checkpoint())."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return Policy.from_checkpoint(checkpoint.get("model", checkpoint)).to(device)
