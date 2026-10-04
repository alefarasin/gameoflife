"""Placeholder simulation: emits plausible-looking metrics until the real world simulator exists.

Replace `FakeSim.step` with the real simulator; the reporting contract (`metrics()`) stays the same.
"""
from __future__ import annotations

import numpy as np

from gol_shared.config import SimConfig


class FakeSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = 0
        self.generation = 0
        self.population = float(cfg.initial_population)
        self.fitness = 1.0

    def step(self, n: int = 50) -> None:
        cfg = self.cfg
        for _ in range(n):
            self.tick += 1
            if self.tick % 100 == 0:
                self.generation += 1
                self.fitness += max(0.0, self.rng.normal(0.05, 0.03)) * (1 - self.fitness / 10)
            growth = 0.002 * (1 - self.population / cfg.max_population)
            self.population += self.population * (growth + self.rng.normal(0, 0.003))
            self.population = float(np.clip(self.population, cfg.min_population, cfg.max_population))

    def metrics(self) -> dict:
        return {
            "tick": self.tick,
            "generation": self.generation,
            "population": int(round(self.population)),
            "avg_energy": float(np.clip(self.cfg.start_energy * (0.8 + 0.1 * self.fitness)
                                        + self.rng.normal(0, 1), 0, self.cfg.max_energy)),
            "avg_fitness": float(self.fitness),
            "max_fitness": float(self.fitness * (1.5 + abs(self.rng.normal(0, 0.1)))),
        }
