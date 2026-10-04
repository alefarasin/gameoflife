"""Simulation configuration shared by backend (validation, presets) and worker (simulation).

Every field carries a `group` and a description so the web UI can build the
settings form directly from the JSON schema.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LEARNING_MODES = ("none", "qlearning", "sarsa", "dqn", "ppo")
SHARED_MODES = ("dqn", "ppo")  # one network shared by the whole population
PER_AGENT_MODES = ("qlearning", "sarsa")  # every creature learns its own weights


def F(default, group: str, description: str, **kw):
    return Field(default, description=description, json_schema_extra={"group": group}, **kw)


class SimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ---- world -------------------------------------------------------------
    width: int = F(96, "World", "Grid width (cells)", ge=16, le=512)
    height: int = F(96, "World", "Grid height (cells)", ge=16, le=512)
    seed: int = F(1, "World", "Random seed (same seed + config = same run)", ge=0)
    wrap: bool = F(True, "World", "Toroidal world (edges wrap around)")
    obstacle_density: float = F(0.03, "World", "Fraction of cells holding an obstacle", ge=0, le=0.5)
    food_max: float = F(1.0, "World", "Max food per cell", gt=0, le=10)
    food_regrow_prob: float = F(0.02, "World", "Per-tick probability that an empty cell grows food", ge=0, le=1)
    food_regrow_amount: float = F(0.5, "World", "Food added when a cell regrows", ge=0, le=10)
    food_energy: float = F(25.0, "World", "Energy gained per unit of food eaten", gt=0, le=500)

    # ---- life layer (Conway-like terrain) -----------------------------------
    life_layer: bool = F(False, "Life layer", "Enable a Game-of-Life terrain layer that boosts food growth")
    life_birth: str = F("3", "Life layer", "Birth neighbour counts (digits 0-8)", pattern=r"^[0-8]*$")
    life_survive: str = F("23", "Life layer", "Survival neighbour counts (digits 0-8)", pattern=r"^[0-8]*$")
    life_density: float = F(0.2, "Life layer", "Initial fraction of live terrain cells", ge=0, le=1)
    life_period: int = F(5, "Life layer", "Terrain updates every N ticks", ge=1, le=1000)
    life_food_boost: float = F(4.0, "Life layer", "Regrowth probability multiplier on live terrain cells", ge=0, le=100)

    # ---- creatures ---------------------------------------------------------
    initial_population: int = F(300, "Creatures", "Starting number of creatures", ge=1, le=100000)
    max_population: int = F(3000, "Creatures", "Hard cap on population (memory is preallocated)", ge=1, le=100000)
    min_population: int = F(20, "Creatures", "Below this, immigrants are injected (0 = allow extinction)", ge=0)
    start_energy: float = F(50.0, "Creatures", "Energy of newborn / initial creatures", gt=0)
    max_energy: float = F(100.0, "Creatures", "Energy cap", gt=0)
    max_age: int = F(1500, "Creatures", "Creatures die of old age after this many ticks (0 = immortal)", ge=0)
    base_cost: float = F(0.15, "Creatures", "Energy burnt per tick (scaled by body size)", ge=0)
    move_cost: float = F(0.10, "Creatures", "Extra energy burnt when moving forward", ge=0)
    vision_cost: float = F(0.01, "Creatures", "Energy per tick per unit of vision range", ge=0)
    brain_cost: float = F(0.0, "Creatures", "Energy per tick per hidden neuron (penalises big brains)", ge=0)
    repro_cost: float = F(10.0, "Creatures", "Extra energy a parent loses on reproduction", ge=0)
    min_repro_age: int = F(30, "Creatures", "Minimum age before reproducing", ge=0)
    hidden_size: int = F(12, "Creatures", "Hidden neurons of the creature brain", ge=2, le=128)

    # ---- interaction -------------------------------------------------------
    predation: bool = F(True, "Interaction", "Allow creatures to attack the one in front")
    attack_damage: float = F(20.0, "Interaction", "Damage of an attack (scaled by attacker size)", ge=0)
    attack_steal: float = F(0.8, "Interaction", "Fraction of dealt damage the attacker gains as energy", ge=0, le=2)
    attack_cost: float = F(0.5, "Interaction", "Energy cost of attacking", ge=0)

    # ---- evolution ---------------------------------------------------------
    mutation_rate: float = F(0.1, "Evolution", "Per-weight probability of mutation in offspring", ge=0, le=1)
    mutation_std: float = F(0.15, "Evolution", "Std-dev of weight mutations", ge=0, le=5)
    gene_mutation_std: float = F(0.08, "Evolution", "Relative std-dev of body / RL hyper-parameter mutations", ge=0, le=1)
    vision_range: tuple[int, int] = F((1, 8), "Evolution", "Evolvable vision range [min, max]")
    size_range: tuple[float, float] = F((0.5, 1.5), "Evolution", "Evolvable body size range [min, max]")
    repro_range: tuple[float, float] = F((0.4, 0.95), "Evolution", "Evolvable reproduction threshold (fraction of max energy)")

    # ---- learning ----------------------------------------------------------
    learning: Literal["none", "qlearning", "sarsa", "dqn", "ppo"] = F(
        "qlearning", "Learning",
        "none: pure neuroevolution | qlearning/sarsa: per-creature online TD learning | "
        "dqn/ppo: shared PyTorch network trained on every creature's experience")
    inherit_learned: bool = F(False, "Learning", "Lamarckian inheritance: children inherit learned (not birth) weights (per-creature modes)")
    gamma: float = F(0.95, "Learning", "Discount factor", ge=0, le=1)
    lr_range: tuple[float, float] = F((0.001, 0.05), "Learning", "Evolvable learning-rate range (per-creature modes)")
    epsilon_range: tuple[float, float] = F((0.01, 0.3), "Learning", "Evolvable exploration-rate range")
    batch_size: int = F(256, "Learning", "DQN minibatch size", ge=8)
    replay_size: int = F(50000, "Learning", "DQN replay buffer size", ge=1000)
    train_every: int = F(4, "Learning", "DQN: gradient step every N ticks", ge=1)
    target_sync: int = F(500, "Learning", "DQN: target network sync every N gradient steps", ge=1)
    shared_lr: float = F(0.0005, "Learning", "Learning rate of the shared network (dqn/ppo)", gt=0)
    shared_hidden: int = F(64, "Learning", "Hidden units of the shared network (dqn/ppo)", ge=8, le=512)
    ppo_rollout: int = F(2048, "Learning", "PPO: samples collected before each update", ge=64)
    ppo_epochs: int = F(4, "Learning", "PPO: epochs per update", ge=1, le=20)
    ppo_clip: float = F(0.2, "Learning", "PPO: clipping range", gt=0, le=1)
    ppo_entropy: float = F(0.01, "Learning", "PPO: entropy bonus", ge=0, le=1)

    # ---- rewards (RL signal; evolution selects only through survival) -------
    r_food: float = F(0.05, "Rewards", "Reward per unit of energy eaten")
    r_survive: float = F(0.01, "Rewards", "Reward per tick alive")
    r_death: float = F(-1.0, "Rewards", "Reward on death")
    r_attack: float = F(0.03, "Rewards", "Reward per unit of damage dealt")
    r_reproduce: float = F(1.0, "Rewards", "Reward on reproduction")
    r_blocked: float = F(-0.02, "Rewards", "Reward when a move is blocked")

    @model_validator(mode="after")
    def _check(self) -> "SimConfig":
        if self.max_population < self.initial_population:
            raise ValueError("max_population must be >= initial_population")
        if self.start_energy > self.max_energy:
            raise ValueError("start_energy must be <= max_energy")
        for name in ("vision_range", "size_range", "repro_range", "lr_range", "epsilon_range"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name}: min must be <= max")
        if self.vision_range[0] < 1 or self.vision_range[1] > 16:
            raise ValueError("vision_range must stay within [1, 16]")
        return self


def default_config() -> dict:
    return SimConfig().model_dump()
