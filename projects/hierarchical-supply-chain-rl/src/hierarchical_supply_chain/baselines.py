"""Classical hierarchical supply-chain policies."""

from __future__ import annotations

import numpy as np


class BaseStockHierarchicalPolicy:
    """Weekly capacity target plus daily base-stock replenishment."""

    def __init__(self, inventory_target: float = 0.38):
        self.inventory_target = inventory_target

    def act(self, obs: np.ndarray) -> np.ndarray:
        inventory, backlog, _, demand, _, week_phase, *_ = obs
        tactical = np.clip(0.45 + 0.65 * backlog + 0.25 * demand - 0.25 * inventory, 0.2, 1.0)
        operational = np.clip(0.35 + 0.9 * backlog + 0.6 * (self.inventory_target - inventory), 0.0, 1.0)
        return np.asarray([tactical, operational], dtype=np.float32)


class ConstantCapacityPolicy:
    """Simple non-hierarchical benchmark with fixed tactical and daily releases."""

    def __init__(self, target: float = 0.65, release: float = 0.55):
        self.target = target
        self.release = release

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray([self.target, self.release], dtype=np.float32)
