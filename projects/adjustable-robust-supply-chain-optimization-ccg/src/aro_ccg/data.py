from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupplyChainInstance:
    fixed_cost: np.ndarray
    reservation_cost: np.ndarray
    capacity_max: np.ndarray
    shipping_cost: np.ndarray
    shortage_penalty: np.ndarray
    nominal_demand: np.ndarray
    demand_deviation: np.ndarray
    max_availability_loss: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "fixed_cost": self.fixed_cost,
            "reservation_cost": self.reservation_cost,
            "capacity_max": self.capacity_max,
            "shipping_cost": self.shipping_cost,
            "shortage_penalty": self.shortage_penalty,
            "nominal_demand": self.nominal_demand,
            "demand_deviation": self.demand_deviation,
            "max_availability_loss": self.max_availability_loss,
        }
        for name, value in arrays.items():
            arr = np.asarray(value, dtype=float)
            object.__setattr__(self, name, arr)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")

        if self.fixed_cost.ndim != 1:
            raise ValueError("fixed_cost must be one-dimensional")
        if self.shipping_cost.ndim != 2:
            raise ValueError("shipping_cost must be two-dimensional")

        n_suppliers = self.fixed_cost.size
        n_markets = self.shortage_penalty.size

        supplier_vectors = (
            self.reservation_cost,
            self.capacity_max,
            self.max_availability_loss,
        )
        if any(v.shape != (n_suppliers,) for v in supplier_vectors):
            raise ValueError("supplier vectors must match fixed_cost length")

        market_vectors = (
            self.nominal_demand,
            self.demand_deviation,
        )
        if any(v.shape != (n_markets,) for v in market_vectors):
            raise ValueError("market vectors must match shortage_penalty length")

        if self.shipping_cost.shape != (n_suppliers, n_markets):
            raise ValueError("shipping_cost shape must be (n_suppliers, n_markets)")

        nonnegative = (
            self.fixed_cost,
            self.reservation_cost,
            self.capacity_max,
            self.shipping_cost,
            self.shortage_penalty,
            self.nominal_demand,
            self.demand_deviation,
            self.max_availability_loss,
        )
        if any(np.any(v < -1e-12) for v in nonnegative):
            raise ValueError("costs, capacities, demands, deviations, and losses must be nonnegative")
        if np.any(self.max_availability_loss > 1.0 + 1e-12):
            raise ValueError("max_availability_loss cannot exceed 1")
        if np.any(self.shortage_penalty <= 0):
            raise ValueError("shortage penalties must be strictly positive")

    @property
    def n_suppliers(self) -> int:
        return int(self.fixed_cost.size)

    @property
    def n_markets(self) -> int:
        return int(self.shortage_penalty.size)


def demo_instance() -> SupplyChainInstance:
    """Return the deterministic synthetic benchmark instance used by tests and the CLI."""
    return SupplyChainInstance(
        fixed_cost=np.array([145.0, 130.0, 120.0]),
        reservation_cost=np.array([2.10, 2.35, 1.95]),
        capacity_max=np.array([105.0, 95.0, 90.0]),
        shipping_cost=np.array(
            [
                [2.2, 3.4, 4.2, 5.0],
                [3.5, 2.1, 3.0, 4.1],
                [4.6, 3.6, 2.0, 2.4],
            ]
        ),
        shortage_penalty=np.array([20.0, 21.0, 22.0, 23.0]),
        nominal_demand=np.array([52.0, 44.0, 48.0, 41.0]),
        demand_deviation=np.array([17.0, 14.0, 16.0, 13.0]),
        max_availability_loss=np.array([0.40, 0.35, 0.45]),
    )
