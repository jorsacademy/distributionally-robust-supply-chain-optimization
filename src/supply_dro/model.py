from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    unit_cost: float = 2.0
    holding_cost: float = 1.0
    shortage_cost: float = 7.0

    def cost(self, order_quantity, demand):
        q = float(order_quantity)
        d = np.asarray(demand, dtype=float)
        return (
            self.unit_cost * q
            + self.holding_cost * np.maximum(q - d, 0.0)
            + self.shortage_cost * np.maximum(d - q, 0.0)
        )


def empirical_demand(seed=7, n=30):
    rng = np.random.default_rng(seed)
    return np.rint(np.clip(rng.normal(50.0, 9.0, n), 20.0, 85.0)).astype(float)


def shifted_demand(seed=77, n=200):
    rng = np.random.default_rng(seed)
    return np.rint(np.clip(rng.normal(61.0, 13.0, n), 20.0, 100.0)).astype(float)
