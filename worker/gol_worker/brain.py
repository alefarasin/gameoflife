"""Batched tiny MLPs (one per creature) with manual backprop for online TD learning."""
from __future__ import annotations

import numpy as np

N_INPUTS = 11
N_ACTIONS = 5


class Brains:
    """Weights of all creatures stored as stacked arrays; forward/TD update are vectorised."""

    FIELDS = ("W1", "b1", "W2", "b2")

    def __init__(self, capacity: int, hidden: int):
        self.hidden = hidden
        self.W1 = np.zeros((capacity, N_INPUTS, hidden), np.float32)
        self.b1 = np.zeros((capacity, hidden), np.float32)
        self.W2 = np.zeros((capacity, hidden, N_ACTIONS), np.float32)
        self.b2 = np.zeros((capacity, N_ACTIONS), np.float32)

    def randomize(self, idx: np.ndarray, rng: np.random.Generator) -> None:
        n = len(idx)
        self.W1[idx] = rng.normal(0, 1 / np.sqrt(N_INPUTS), (n, N_INPUTS, self.hidden))
        self.b1[idx] = 0
        self.W2[idx] = rng.normal(0, 1 / np.sqrt(self.hidden), (n, self.hidden, N_ACTIONS))
        self.b2[idx] = 0

    def copy_mutated(self, dst: np.ndarray, src: np.ndarray, rate: float, std: float,
                     rng: np.random.Generator, source: "Brains | None" = None) -> None:
        """Copy brains `src` -> slots `dst`, perturbing a random subset of weights."""
        source = source or self
        for f in self.FIELDS:
            w = getattr(source, f)[src].copy()
            mask = rng.random(w.shape) < rate
            w += (rng.normal(0, std, w.shape) * mask).astype(np.float32)
            getattr(self, f)[dst] = w

    def forward(self, idx: np.ndarray, x: np.ndarray):
        h = np.tanh(np.einsum("ni,nih->nh", x, self.W1[idx]) + self.b1[idx])
        out = np.einsum("nh,nha->na", h, self.W2[idx]) + self.b2[idx]
        return h, out

    def td_update(self, idx: np.ndarray, x: np.ndarray, a: np.ndarray, target: np.ndarray,
                  lr: np.ndarray) -> float:
        """One semi-gradient step on 0.5*(Q(x,a)-target)^2 (Huber-clipped) for each creature."""
        if len(idx) == 0:
            return 0.0
        h, q = self.forward(idx, x)
        rows = np.arange(len(idx))
        err = np.clip(q[rows, a] - target, -1.0, 1.0).astype(np.float32)
        d_out = np.zeros_like(q)
        d_out[rows, a] = err
        lr = lr.astype(np.float32)[:, None]
        W2 = self.W2[idx]
        dh = np.einsum("na,nha->nh", d_out, W2) * (1 - h * h)
        self.W2[idx] = W2 - lr[:, :, None] * h[:, :, None] * d_out[:, None, :]
        self.b2[idx] -= lr * d_out
        self.W1[idx] -= lr[:, :, None] * x[:, :, None] * dh[:, None, :]
        self.b1[idx] -= lr * dh
        return float(np.abs(err).mean())

    def state(self) -> dict:
        return {f"brain_{f}": getattr(self, f) for f in self.FIELDS}

    def load(self, data) -> None:
        for f in self.FIELDS:
            getattr(self, f)[...] = data[f"brain_{f}"]
