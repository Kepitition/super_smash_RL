"""PPO for the recurrent policy, with a KL leash to the imitation policy (the "teacher").

A rollout is `rollout` consecutive frames of each of n games, cut into chunks of `chunk`
frames. Acting stores the policy's LSTM state at every chunk start, so the PPO minibatches
(sets of chunks) start where acting was; the teacher and the value network run over the
rollout once per iteration, carrying their own state from one rollout to the next.
Games end mid-chunk: the reset flags restart the LSTMs on a game's first frame.

Loss per frame: clipped PPO surrogate + kl_coef * KL(policy || teacher) - entropy_coef *
entropy, over the four controller heads (each conditioned on the taken earlier classes).
"""

import collections

import numpy as np
import torch
from torch import nn

from smash_rl.codec import encode_state
from smash_rl.model import Policy


class ValueNet(nn.Module):
    """A Policy core (same inputs, own weights) with a linear value head, starting at 0."""

    def __init__(self, **config):
        super().__init__()
        self.core = Policy(**config)
        self.head = nn.Linear(self.core.config["width"], 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_policy(cls, policy: Policy) -> "ValueNet":
        """Starts from the imitation policy's core, which already understands the game."""
        value = cls(**policy.config)
        value.core.load_state_dict(policy.state_dict())
        return value

    def initial_state(self, batch: int, device) -> list:
        return self.core.initial_state(batch, device)

    def forward(self, inputs, state, reset):
        out, state = self.core(inputs, state, reset)
        return self.head(out)[..., 0], state

    def checkpoint(self) -> dict:
        return {"config": dict(self.core.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "ValueNet":
        value = cls(**checkpoint["config"])
        value.load_state_dict(checkpoint["state_dict"])
        return value


def gae(rewards, values, dones, gamma: float, lam: float) -> np.ndarray:
    """Advantages (n, T) from rewards and dones (n, T) and values (n, T + 1)."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros(rewards.shape[0], dtype=np.float32)
    for t in reversed(range(rewards.shape[1])):
        live = 1.0 - dones[:, t].astype(np.float32)
        delta = rewards[:, t] + gamma * values[:, t + 1] * live - values[:, t]
        last = delta + gamma * lam * live * last
        advantages[:, t] = last
    return advantages


class Rollout:
    """Frames of n games over `length` steps (plus the next frame, to bootstrap from)."""

    def __init__(self, n: int, length: int, chunk: int):
        if length % chunk:
            raise ValueError(f"rollout {length} is not a multiple of chunk {chunk}")
        self.n, self.length, self.chunk = n, length, chunk
        self.players, self.stage, self.first, self.prev_action = [], [], [], []
        self.actions, self.logp, self.rewards, self.dones = [], [], [], []
        self.chunk_states = []  # the acting policy's LSTM state at each chunk start

    def add_frame(self, players, stage, first, prev_action):
        self.players.append(players.copy())
        self.stage.append(np.asarray(stage).copy())
        self.first.append(np.asarray(first).copy())
        self.prev_action.append(np.asarray(prev_action).copy())

    def add_step(self, actions, logp, rewards, dones):
        self.actions.append(actions.copy())
        self.logp.append(logp.copy())
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones).copy())

    def tensors(self, device) -> dict:
        """(n, T + 1, ...) model inputs and reset flags; (n, T, ...) actions, log-probs, rewards, dones."""
        players = np.stack(self.players, axis=1)  # (n, T + 1, 2)
        inputs = encode_state(players, np.stack(self.stage, axis=1))
        inputs["prev_action"] = np.stack(self.prev_action, axis=1).astype(np.int64)
        out = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
        out["reset"] = torch.from_numpy(np.stack(self.first, axis=1)).to(device)
        out["actions"] = torch.from_numpy(np.stack(self.actions, axis=1).astype(np.int64)).to(device)
        out["logp"] = torch.from_numpy(np.stack(self.logp, axis=1).astype(np.float32)).to(device)
        out["rewards"] = np.stack(self.rewards, axis=1)
        out["dones"] = np.stack(self.dones, axis=1)
        return out


INPUT_KEYS = ("numeric", "action", "character", "jumps", "stage", "prev_action")


def _frames(batch: dict, start: int, end: int) -> dict:
    return {k: batch[k][:, start:end] for k in INPUT_KEYS}


def _stack_states(states: list) -> list:
    """Per-chunk states [chunk][layer](h, c) of (1, n, w) -> [layer](h, c) of (chunks, n, w)."""
    return [(torch.cat([s[layer][0] for s in states]), torch.cat([s[layer][1] for s in states]))
            for layer in range(len(states[0]))]


@torch.no_grad()
def run_over_rollout(model, batch: dict, state: list, chunk: int, length: int):
    """Runs `model` over frames [0, length) chunk by chunk: (outputs per chunk, states at
    chunk starts, state after the last frame)."""
    outputs, starts = [], []
    for start in range(0, length, chunk):
        starts.append([(h.clone(), c.clone()) for h, c in state])
        out, state = model(_frames(batch, start, start + chunk), state, batch["reset"][:, start:start + chunk])
        outputs.append(out)
    return outputs, starts, state


def head_log_probs(logps: list, actions) -> torch.Tensor:
    """log-probability of the taken controllers, summed over the four heads."""
    return sum(lp.gather(-1, actions[..., k:k + 1])[..., 0] for k, lp in enumerate(logps))


def kl_and_entropy(logps: list, teacher_logps: list):
    """Per-frame KL(policy || teacher) and policy entropy, summed over the heads."""
    kl = sum((lp.exp() * (lp - tlp)).sum(-1) for lp, tlp in zip(logps, teacher_logps))
    entropy = sum(-(lp.exp() * lp).sum(-1) for lp in logps)
    return kl, entropy


def minibatches(n: int, chunks: int, count: int, rng: np.random.Generator):
    """Shuffled (game index, chunk index) arrays, `count` minibatches."""
    pairs = np.stack(np.meshgrid(np.arange(n), np.arange(chunks), indexing="ij"), -1).reshape(-1, 2)
    rng.shuffle(pairs)
    return [(torch.from_numpy(p[:, 0]), torch.from_numpy(p[:, 1])) for p in np.array_split(pairs, count)]


def select(batch_tensor, games, chunk_ids, chunk: int):
    """(B, chunk, ...) frames of the chosen (game, chunk) pairs from an (n, T, ...) tensor."""
    n, length = batch_tensor.shape[:2]
    view = batch_tensor[:, :length - length % chunk].reshape(n, length // chunk, chunk, *batch_tensor.shape[2:])
    return view[games.to(view.device), chunk_ids.to(view.device)]


def select_state(stacked: list, games, chunk_ids) -> list:
    """Start states (1, B, w) of the chosen pairs from [layer](h, c) of (chunks, n, w)."""
    return [(h[chunk_ids.to(h.device), games.to(h.device)][None].contiguous(),
             c[chunk_ids.to(c.device), games.to(c.device)][None].contiguous()) for h, c in stacked]


class Learner:
    """Holds the policy, teacher and value network and does one PPO update per rollout."""

    def __init__(self, policy: Policy, teacher: Policy, value: ValueNet, n: int, device, *, lr: float = 3e-5,
                 value_lr: float = 1e-4, gamma: float = 0.997, lam: float = 0.98, clip: float = 0.2,
                 kl_coef: float = 0.5, entropy_coef: float = 0.0, epochs: int = 2, minibatches: int = 4,
                 max_grad_norm: float = 1.0, seed: int = 0):
        self.policy, self.teacher, self.value = policy, teacher, value
        self.device = torch.device(device)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.kl_coef, self.entropy_coef = kl_coef, entropy_coef
        self.epochs, self.minibatches, self.max_grad_norm = epochs, minibatches, max_grad_norm
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(value.parameters(), lr=value_lr)
        # Carried from one rollout to the next, like the acting policy's state.
        self.teacher_state = teacher.initial_state(n, self.device)
        self.value_state = value.initial_state(n, self.device)
        self.rng = np.random.default_rng(seed)

    def state_dict(self) -> dict:
        return {"policy_opt": self.policy_opt.state_dict(), "value_opt": self.value_opt.state_dict()}

    def load_state_dict(self, state: dict):
        self.policy_opt.load_state_dict(state["policy_opt"])
        self.value_opt.load_state_dict(state["value_opt"])

    def update(self, rollout: Rollout, train_policy: bool = True) -> dict:
        """One PPO update (the value network only while `train_policy` is False); returns statistics."""
        length, chunk, n = rollout.length, rollout.chunk, rollout.n
        batch = rollout.tensors(self.device)
        with torch.no_grad():
            outs, _, self.teacher_state = run_over_rollout(self.teacher, batch, self.teacher_state, chunk, length)
            per_chunk = [self.teacher.log_probs(out, batch["actions"][:, s:s + chunk])
                         for out, s in zip(outs, range(0, length, chunk))]
            teacher_logps = [torch.cat(parts, dim=1) for parts in zip(*per_chunk)]
            outs, value_starts, self.value_state = run_over_rollout(self.value, batch, self.value_state, chunk, length)
            bootstrap, _ = self.value(_frames(batch, length, length + 1), self.value_state,
                                      batch["reset"][:, length:length + 1])
            values = torch.cat(outs + [bootstrap], dim=1).float().cpu().numpy()
        advantages = gae(batch["rewards"], values, batch["dones"], self.gamma, self.lam)
        returns = advantages + values[:, :length]
        norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_t = torch.from_numpy(norm_adv.astype(np.float32)).to(self.device)
        ret_t = torch.from_numpy(returns.astype(np.float32)).to(self.device)
        policy_states, value_states = _stack_states(rollout.chunk_states), _stack_states(value_starts)
        inputs_all = {k: batch[k][:, :length] for k in INPUT_KEYS}
        reset_all = batch["reset"][:, :length]

        stats = collections.defaultdict(list)
        self.policy.train()
        self.value.train()
        for _ in range(self.epochs):
            for games, chunk_ids in minibatches(n, length // chunk, self.minibatches, self.rng):
                def sel(x):
                    return select(x, games, chunk_ids, chunk)

                inputs = {k: sel(v) for k, v in inputs_all.items()}
                reset, actions = sel(reset_all), sel(batch["actions"])
                v, _ = self.value(inputs, select_state(value_states, games, chunk_ids), reset)
                value_loss = 0.5 * (v - sel(ret_t)).pow(2).mean()
                self.value_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_opt.step()

                with torch.set_grad_enabled(train_policy):
                    out, _ = self.policy(inputs, select_state(policy_states, games, chunk_ids), reset)
                    logps = self.policy.log_probs(out, actions)
                    new_logp = head_log_probs(logps, actions)
                    old_logp = sel(batch["logp"])
                    ratio = (new_logp - old_logp).exp()
                    adv = sel(adv_t)
                    pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
                    kl, entropy = kl_and_entropy(logps, [sel(t) for t in teacher_logps])
                    loss = pg_loss + self.kl_coef * kl.mean() - self.entropy_coef * entropy.mean()
                if train_policy:
                    self.policy_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    stats["grad_norm"].append(
                        float(nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)))
                    self.policy_opt.step()
                stats["value_loss"].append(value_loss.item())
                stats["pg_loss"].append(pg_loss.item())
                stats["teacher_kl"].append(kl.mean().item())
                stats["entropy"].append(entropy.mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > self.clip).float().mean().item())
                stats["approx_kl"].append((old_logp - new_logp).mean().item())
        self.policy.eval()
        self.value.eval()
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        residual = returns - values[:, :length]
        out["explained_variance"] = float(1 - residual.var() / max(returns.var(), 1e-8))
        out["reward_per_frame"] = float(batch["rewards"].mean())
        return out
