from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SupplyChainInstance:
    facility_names: tuple[str, ...]
    customer_names: tuple[str, ...]
    fixed_cost: np.ndarray
    capacity: np.ndarray
    nominal_demand: np.ndarray
    demand_deviation: np.ndarray
    shipping_cost: np.ndarray

    def __post_init__(self) -> None:
        f = len(self.facility_names)
        c = len(self.customer_names)
        arrays = {
            "fixed_cost": (self.fixed_cost, (f,)),
            "capacity": (self.capacity, (f,)),
            "nominal_demand": (self.nominal_demand, (c,)),
            "demand_deviation": (self.demand_deviation, (c,)),
            "shipping_cost": (self.shipping_cost, (f, c)),
        }
        for name, (array, shape) in arrays.items():
            if np.asarray(array).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {np.asarray(array).shape}")
        if np.any(self.fixed_cost < 0):
            raise ValueError("fixed_cost must be nonnegative")
        if np.any(self.capacity <= 0):
            raise ValueError("capacity must be strictly positive")
        if np.any(self.nominal_demand < 0):
            raise ValueError("nominal_demand must be nonnegative")
        if np.any(self.demand_deviation < 0):
            raise ValueError("demand_deviation must be nonnegative")
        if np.any(self.shipping_cost < 0):
            raise ValueError("shipping_cost must be nonnegative")

    @property
    def n_facilities(self) -> int:
        return len(self.facility_names)

    @property
    def n_customers(self) -> int:
        return len(self.customer_names)


def euclidean_shipping_cost(
    facility_xy: Sequence[Sequence[float]],
    customer_xy: Sequence[Sequence[float]],
    cost_per_distance_unit: float = 0.06,
) -> np.ndarray:
    facility_xy_arr = np.asarray(facility_xy, dtype=float)
    customer_xy_arr = np.asarray(customer_xy, dtype=float)
    if facility_xy_arr.ndim != 2 or facility_xy_arr.shape[1] != 2:
        raise ValueError("facility_xy must have shape (n_facilities, 2)")
    if customer_xy_arr.ndim != 2 or customer_xy_arr.shape[1] != 2:
        raise ValueError("customer_xy must have shape (n_customers, 2)")
    if cost_per_distance_unit < 0:
        raise ValueError("cost_per_distance_unit must be nonnegative")
    delta = facility_xy_arr[:, None, :] - customer_xy_arr[None, :, :]
    distance = np.sqrt(np.sum(delta**2, axis=2))
    return cost_per_distance_unit * distance


def demo_instance() -> SupplyChainInstance:
    """Return a deterministic synthetic capacitated facility-location instance."""
    facility_names = ("North", "Central", "South", "East", "West")
    customer_names = tuple(f"Zone-{i:02d}" for i in range(1, 13))

    facility_xy = np.array(
        [
            [45.0, 86.0],
            [52.0, 52.0],
            [48.0, 15.0],
            [86.0, 50.0],
            [14.0, 48.0],
        ],
        dtype=float,
    )
    customer_xy = np.array(
        [
            [24.0, 79.0],
            [45.0, 92.0],
            [69.0, 83.0],
            [82.0, 67.0],
            [88.0, 42.0],
            [70.0, 25.0],
            [48.0, 8.0],
            [25.0, 18.0],
            [9.0, 35.0],
            [13.0, 61.0],
            [40.0, 58.0],
            [62.0, 48.0],
        ],
        dtype=float,
    )

    fixed_cost = np.array([1220.0, 1500.0, 1180.0, 1110.0, 1090.0])
    capacity = np.array([330.0, 390.0, 325.0, 300.0, 295.0])
    nominal_demand = np.array(
        [78.0, 84.0, 73.0, 88.0, 82.0, 75.0, 81.0, 69.0, 76.0, 74.0, 96.0, 91.0]
    )
    deviation_fraction = np.array(
        [0.23, 0.20, 0.27, 0.18, 0.25, 0.22, 0.24, 0.30, 0.26, 0.21, 0.19, 0.28]
    )
    demand_deviation = nominal_demand * deviation_fraction
    shipping_cost = euclidean_shipping_cost(facility_xy, customer_xy, 0.055)

    return SupplyChainInstance(
        facility_names=facility_names,
        customer_names=customer_names,
        fixed_cost=fixed_cost,
        capacity=capacity,
        nominal_demand=nominal_demand,
        demand_deviation=demand_deviation,
        shipping_cost=shipping_cost,
    )
