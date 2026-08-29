from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class NetworkData:
    reservation_cost: np.ndarray
    transport_cost: np.ndarray
    supplier_capacity: np.ndarray
    shortage_cost: np.ndarray


def default_network() -> NetworkData:
    return NetworkData(
        reservation_cost=np.array([1.8, 2.2], dtype=float),
        transport_cost=np.array([[1.0, 2.4], [2.1, 1.1]], dtype=float),
        supplier_capacity=np.array([80.0, 80.0], dtype=float),
        shortage_cost=np.array([9.0, 11.0], dtype=float),
    )


def demand_scenarios(seed: int = 7, n: int = 20, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.array([48.0 + shift, 42.0 + 0.8 * shift])
    covariance = np.array([[90.0, 28.0], [28.0, 72.0]])
    scenarios = rng.multivariate_normal(mean, covariance, size=n)
    return np.rint(np.clip(scenarios, 10.0, 90.0)).astype(float)


def disruption_scenarios() -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Finite stress set for classical robust optimization.

    Availability values multiply the usable reserved capacity of each supplier.
    """
    return [
        (np.array([55.0, 48.0]), np.array([1.0, 1.0]), "high_demand"),
        (np.array([62.0, 52.0]), np.array([1.0, 1.0]), "demand_surge"),
        (np.array([52.0, 46.0]), np.array([0.35, 1.0]), "supplier_0_disruption"),
        (np.array([52.0, 46.0]), np.array([1.0, 0.35]), "supplier_1_disruption"),
        (np.array([60.0, 50.0]), np.array([0.60, 0.70]), "joint_stress"),
    ]


def recourse_cost(
    reserved_capacity: np.ndarray,
    demand: np.ndarray,
    data: NetworkData,
    availability: np.ndarray | None = None,
) -> float:
    """Solve shipment and shortage recourse for fixed reserved capacity."""
    reserved = np.asarray(reserved_capacity, dtype=float)
    demand = np.asarray(demand, dtype=float)
    if availability is None:
        availability = np.ones_like(data.supplier_capacity, dtype=float)
    availability = np.asarray(availability, dtype=float)
    if availability.shape != data.supplier_capacity.shape:
        raise ValueError("availability must match supplier capacity dimension")
    if np.any((availability < 0.0) | (availability > 1.0)):
        raise ValueError("availability values must be between 0 and 1")

    n_suppliers, n_markets = data.transport_cost.shape
    n_ship = n_suppliers * n_markets
    objective = np.concatenate([data.transport_cost.reshape(-1), data.shortage_cost])
    constraints = []
    rhs = []

    for supplier in range(n_suppliers):
        row = np.zeros(n_ship + n_markets)
        start = supplier * n_markets
        row[start:start + n_markets] = 1.0
        constraints.append(row)
        usable = min(reserved[supplier], data.supplier_capacity[supplier]) * availability[supplier]
        rhs.append(usable)

    for market in range(n_markets):
        row = np.zeros(n_ship + n_markets)
        for supplier in range(n_suppliers):
            row[supplier * n_markets + market] = -1.0
        row[n_ship + market] = -1.0
        constraints.append(row)
        rhs.append(-demand[market])

    result = linprog(
        objective,
        A_ub=np.asarray(constraints),
        b_ub=np.asarray(rhs),
        bounds=[(0.0, None)] * (n_ship + n_markets),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


def first_stage_cost(reserved_capacity: np.ndarray, data: NetworkData) -> float:
    return float(np.asarray(reserved_capacity, dtype=float) @ data.reservation_cost)
