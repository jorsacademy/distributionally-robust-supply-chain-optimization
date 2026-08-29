from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .network import NetworkData, recourse_cost


def worst_case_joint_recourse(
    reserved_capacity: np.ndarray,
    demand_scenarios: np.ndarray,
    availability_scenarios: np.ndarray,
    data: NetworkData,
    epsilon: float,
    *,
    demand_scale: float = 20.0,
    availability_weight: float = 4.0,
) -> dict[str, object]:
    """Finite-support Wasserstein adversary over joint demand/availability states.

    Demand distances are normalized by ``demand_scale`` while availability loss is
    weighted separately so both uncertainty components contribute to the transport metric.
    """
    demand = np.asarray(demand_scenarios, dtype=float)
    availability = np.asarray(availability_scenarios, dtype=float)
    if demand.ndim != 2 or availability.ndim != 2 or len(demand) != len(availability):
        raise ValueError("demand and availability scenarios must be aligned 2-D arrays")
    n = len(demand)
    if n == 0:
        raise ValueError("at least one joint scenario is required")

    costs = np.asarray(
        [
            recourse_cost(reserved_capacity, demand[i], data, availability=availability[i])
            for i in range(n)
        ],
        dtype=float,
    )
    demand_distance = np.abs(demand[:, None, :] - demand[None, :, :]).sum(axis=2) / demand_scale
    availability_distance = np.abs(
        availability[:, None, :] - availability[None, :, :]
    ).sum(axis=2)
    distance = demand_distance + availability_weight * availability_distance

    source_probability = np.full(n, 1.0 / n)
    a_eq = np.zeros((n, n * n), dtype=float)
    for source in range(n):
        a_eq[source, source * n : (source + 1) * n] = 1.0

    result = linprog(
        -np.tile(costs, n),
        A_ub=distance.reshape(1, -1),
        b_ub=np.array([float(epsilon)]),
        A_eq=a_eq,
        b_eq=source_probability,
        bounds=[(0.0, None)] * (n * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    plan = result.x.reshape(n, n)
    return {
        "worst_case_recourse": float(-result.fun),
        "adversarial_probability": plan.sum(axis=0),
        "transport_used": float(np.sum(plan * distance)),
        "scenario_cost": costs,
    }
