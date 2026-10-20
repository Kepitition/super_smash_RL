"""Imitation-learning batches: stateful unrolls through whole games.

Each of the batch's rows walks through one trajectory (one player of one game) in
consecutive chunks, so the model's LSTM state can be carried from one batch to the
next exactly as when it plays a live game. A row that reaches the end of its
trajectory starts a random new one and is flagged `reset`.
"""

import os
import queue
import threading

import numpy as np
import torch

from smash_rl.codec import NEUTRAL, controller_classes, encode_state
from smash_rl.replays import read_index


def load_index(data_dir: str) -> list:
    return read_index(os.path.join(data_dir, "index.jsonl"))


def trajectory_list(index: list, split: str, data_dir: str) -> list:
    """(path, slot) for every kept player of every game in a split (a ditto gives two)."""
    return [(os.path.join(data_dir, e["path"]), slot)
            for e in index if e.get("split") == split for slot in e["slots"]]


def load_trajectory(path: str, slot: int):
    """(raw frames (T, 2) with the trajectory's player first, stage index)."""
    with np.load(path) as data:
        players = data["players"]
        stage = int(data["stage"])
    if slot == 1:
        players = players[:, ::-1]
    return np.ascontiguousarray(players), stage


class UnrollBatcher:
    """Batches of `unroll` frames: inputs at frames t, labels at t + delay."""

    def __init__(self, trajectories, batch_size: int, unroll: int, delay: int, seed: int, loader=load_trajectory):
        self.trajectories = list(trajectories)
        self.batch_size, self.unroll, self.delay = batch_size, unroll, delay
        self.rng = np.random.default_rng(seed)
        self.loader = loader
        self.readers = [None] * batch_size  # [features, classes, position]

    def _new_reader(self, i: int):
        for _ in range(1000):
            path, slot = self.trajectories[self.rng.integers(len(self.trajectories))]
            players, stage = self.loader(path, slot)
            if len(players) >= self.unroll + self.delay:
                self.readers[i] = [encode_state(players, stage), controller_classes(players[:, 0]), 0]
                return
        raise RuntimeError("no trajectory is long enough for unroll + delay frames")

    def next_batch(self) -> dict:
        U, d = self.unroll, self.delay
        reset = np.zeros(self.batch_size, dtype=bool)
        rows = []
        for i in range(self.batch_size):
            reader = self.readers[i]
            if reader is None or reader[2] + U + d > len(reader[1]):
                self._new_reader(i)
                reset[i] = True
                reader = self.readers[i]
            features, classes, p = reader
            prev = classes[p + d - 1:p + U + d - 1].copy()
            if p == 0:
                prev[0] = NEUTRAL  # live, the first frame has no previous decision either
            rows.append(({k: v[p:p + U] for k, v in features.items()}, prev, classes[p + d:p + U + d]))
            reader[2] = p + U
        batch = {k: np.stack([r[0][k] for r in rows]) for k in rows[0][0]}
        batch["prev_action"] = np.stack([r[1] for r in rows])
        batch["target"] = np.stack([r[2] for r in rows])
        batch["reset"] = reset
        return batch


class Prefetcher:
    """Builds batches in a background thread so the GPU doesn't wait for loading."""

    def __init__(self, batcher: UnrollBatcher, depth: int = 4):
        self.queue = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, args=(batcher,), daemon=True)
        self.thread.start()

    def _run(self, batcher):
        try:
            while True:
                self.queue.put(batcher.next_batch())
        except BaseException as e:  # surface loader errors in the training loop
            self.queue.put(e)

    def next(self) -> dict:
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item


def to_torch(batch: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in batch.items()}
