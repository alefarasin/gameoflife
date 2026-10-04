"""Shared-network learners (DQN, PPO) trained on the pooled experience of every creature.

PyTorch is imported lazily so the rest of the simulator works without it.
"""
from __future__ import annotations

import io

import numpy as np

from .brain import N_ACTIONS, N_INPUTS


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("learning mode dqn/ppo needs PyTorch (pip install torch)") from e


def _mlp(torch, hidden, out):
    nn = torch.nn
    return nn.Sequential(nn.Linear(N_INPUTS, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class DQN:
    name = "dqn"

    def __init__(self, cfg, seed: int):
        torch = _torch()
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.t, self.cfg = torch, cfg
        self.q = _mlp(torch, cfg.shared_hidden, N_ACTIONS)
        self.target = _mlp(torch, cfg.shared_hidden, N_ACTIONS)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.shared_lr)
        n = cfg.replay_size
        self.s = np.zeros((n, N_INPUTS), np.float32)
        self.s2 = np.zeros((n, N_INPUTS), np.float32)
        self.a = np.zeros(n, np.int64)
        self.r = np.zeros(n, np.float32)
        self.d = np.zeros(n, np.float32)
        self.ptr = 0
        self.size = 0
        self.steps = 0
        self.ticks = 0
        self.loss = 0.0
        self.rng = np.random.default_rng(seed + 1)

    def act(self, obs: np.ndarray, eps: np.ndarray):
        with self.t.no_grad():
            q = self.q(self.t.from_numpy(obs)).numpy()
        greedy = q.argmax(1)
        explore = self.rng.random(len(obs)) < eps
        rand = self.rng.integers(0, N_ACTIONS, len(obs))
        return np.where(explore, rand, greedy).astype(np.int64), None

    def push(self, s, a, aux, r, s2, done):
        n = len(s)
        if n == 0:
            return
        idx = (self.ptr + np.arange(n)) % len(self.a)
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx] = s, a, r, s2, done
        self.ptr = int((self.ptr + n) % len(self.a))
        self.size = min(len(self.a), self.size + n)

    def train(self):
        self.ticks += 1
        cfg = self.cfg
        if self.size < cfg.batch_size * 4 or self.ticks % cfg.train_every:
            return
        t = self.t
        b = self.rng.integers(0, self.size, cfg.batch_size)
        s, s2 = t.from_numpy(self.s[b]), t.from_numpy(self.s2[b])
        a, r, d = t.from_numpy(self.a[b]), t.from_numpy(self.r[b]), t.from_numpy(self.d[b])
        with t.no_grad():
            y = r + cfg.gamma * (1 - d) * self.target(s2).max(1).values
        q = self.q(s).gather(1, a[:, None])[:, 0]
        loss = t.nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.loss = float(loss)
        self.steps += 1
        if self.steps % cfg.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())

    def state_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.t.save({"q": self.q.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict()}, buf)
        return buf.getvalue()

    def load_bytes(self, data: bytes) -> None:
        d = self.t.load(io.BytesIO(data))
        self.q.load_state_dict(d["q"])
        self.target.load_state_dict(d["target"])
        self.opt.load_state_dict(d["opt"])


class PPO:
    """Clipped-surrogate PPO with one-step bootstrapped advantages.

    Trajectories of different creatures are interleaved, so instead of GAE we use
    A = r + gamma * V(s') - V(s) (GAE with lambda = 0) which only needs a single transition.
    """
    name = "ppo"

    def __init__(self, cfg, seed: int):
        torch = _torch()
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.t, self.cfg = torch, cfg
        self.pi = _mlp(torch, cfg.shared_hidden, N_ACTIONS)
        self.v = _mlp(torch, cfg.shared_hidden, 1)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=cfg.shared_lr)
        self.buf: list[tuple] = []
        self.count = 0
        self.updates = 0
        self.loss = 0.0
        self.gen = torch.Generator().manual_seed(seed + 2)

    def act(self, obs: np.ndarray, eps: np.ndarray):
        t = self.t
        with t.no_grad():
            logits = self.pi(t.from_numpy(obs))
            dist = t.distributions.Categorical(logits=logits)
            a = t.multinomial(t.softmax(logits, 1), 1, generator=self.gen)[:, 0]
            logp = dist.log_prob(a)
        return a.numpy().astype(np.int64), logp.numpy().astype(np.float32)

    def push(self, s, a, aux, r, s2, done):
        if len(s) == 0:
            return
        self.buf.append((s, a, aux, r, s2, done))
        self.count += len(s)

    def train(self):
        if self.count < self.cfg.ppo_rollout:
            return
        t, cfg = self.t, self.cfg
        s, a, lp, r, s2, d = (np.concatenate([b[i] for b in self.buf]) for i in range(6))
        self.buf, self.count = [], 0
        s, s2 = t.from_numpy(s), t.from_numpy(s2)
        a, lp_old = t.from_numpy(a), t.from_numpy(lp)
        r, d = t.from_numpy(r.astype(np.float32)), t.from_numpy(d.astype(np.float32))
        with t.no_grad():
            ret = r + cfg.gamma * (1 - d) * self.v(s2)[:, 0]
            adv = ret - self.v(s)[:, 0]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(a)
        mb = max(64, n // 4)
        for _ in range(cfg.ppo_epochs):
            perm = t.randperm(n, generator=self.gen)
            for i in range(0, n, mb):
                j = perm[i:i + mb]
                dist = t.distributions.Categorical(logits=self.pi(s[j]))
                ratio = (dist.log_prob(a[j]) - lp_old[j]).exp()
                surr = t.min(ratio * adv[j], ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv[j])
                v_loss = (self.v(s[j])[:, 0] - ret[j]).pow(2).mean()
                loss = -surr.mean() + 0.5 * v_loss - cfg.ppo_entropy * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), 1.0)
                self.opt.step()
                self.loss = float(loss)
        self.updates += 1

    def state_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.t.save({"pi": self.pi.state_dict(), "v": self.v.state_dict(), "opt": self.opt.state_dict()}, buf)
        return buf.getvalue()

    def load_bytes(self, data: bytes) -> None:
        d = self.t.load(io.BytesIO(data))
        self.pi.load_state_dict(d["pi"])
        self.v.load_state_dict(d["v"])
        self.opt.load_state_dict(d["opt"])


def make_learner(cfg, seed: int):
    return {"dqn": DQN, "ppo": PPO}[cfg.learning](cfg, seed)
