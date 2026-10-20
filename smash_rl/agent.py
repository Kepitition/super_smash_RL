"""Runs a Policy live: one LSTM state and previous decision per game, one batched forward per frame.

On a GPU each frame is a single CUDA graph replay: a step of this small batch is mostly
kernel-launch overhead in eager mode (~2.3 ms, vs ~0.7 ms replayed). The graph reads the
model's parameters where they are, so training the model or loading other weights into it
(load_state_dict copies) keeps acting with the current weights. model.to() moves the LSTM
weights (cuDNN re-packs them) even on the same device; the graph is then recorded again.
"""

import numpy as np
import torch

from smash_rl.codec import NEUTRAL, encode_state
from smash_rl.state import PLAYER_DTYPE


class Agent:
    def __init__(self, model, n: int, device, temperature: float = 1.0, cuda_graph: bool = True):
        self.device = torch.device(device)
        if next(model.parameters()).device != self.device:
            model.to(self.device)
        self.model = model.eval()
        self.n, self.temperature = n, temperature
        self.state = model.initial_state(n, self.device)
        self.prev = np.tile(NEUTRAL, (n, 1))
        self.logp = np.zeros(n, dtype=np.float32)  # log-probability of the last decisions
        inputs = encode_state(np.zeros((n, 1, 2), dtype=PLAYER_DTYPE), np.zeros((n, 1), dtype=np.int64))
        inputs["prev_action"] = self.prev[:, None]
        self._inputs = {k: torch.from_numpy(v).to(self.device) for k, v in inputs.items()}
        self._no_reset = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._actions = torch.zeros((n, 4), dtype=torch.int64, device=self.device)
        self._logp = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._graph = None
        if cuda_graph and self.device.type == "cuda":
            self._capture()

    @torch.no_grad()
    def _forward(self):
        """One frame on the fixed input/output tensors (what the CUDA graph records)."""
        out, new_state = self.model(self._inputs, self.state, self._no_reset)
        actions, logp = self.model.sample_with_logp(out[:, 0], self.temperature)
        for (h, c), (new_h, new_c) in zip(self.state, new_state):
            h.copy_(new_h)
            c.copy_(new_c)
        self._actions.copy_(actions)
        self._logp.copy_(logp)

    def _weights(self) -> list:
        return [p.data_ptr() for p in self.model.parameters()]

    def _capture(self):
        self._graph_weights = self._weights()
        saved = [(h.clone(), c.clone()) for h, c in self.state]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # warm-up (cuDNN picks its kernels) outside the graph
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._forward()
        for (h, c), (h0, c0) in zip(self.state, saved):
            h.copy_(h0)
            c.copy_(c0)

    def reset(self, i: int):
        """Game i ended: its next frame starts a new game with a fresh state."""
        for h, c in self.state:
            h[:, i] = 0
            c[:, i] = 0
        self.prev[i] = NEUTRAL

    def state_copy(self) -> list:
        return [(h.clone(), c.clone()) for h, c in self.state]

    @torch.no_grad()
    def act(self, players, stage) -> np.ndarray:
        """Controller classes (n, 4) for raw frames (n, 2) [agent, opponent] and stage indices (n,)."""
        inputs = encode_state(players[:, None], np.asarray(stage)[:, None])
        inputs["prev_action"] = self.prev[:, None]
        for k, v in inputs.items():
            self._inputs[k].copy_(torch.from_numpy(np.ascontiguousarray(v)))
        if self._graph is not None:
            if self._weights() != self._graph_weights:
                self._capture()
            self._graph.replay()
        else:
            self._forward()
        self.prev = self._actions.cpu().numpy()
        self.logp = self._logp.cpu().numpy()
        return self.prev.copy()
