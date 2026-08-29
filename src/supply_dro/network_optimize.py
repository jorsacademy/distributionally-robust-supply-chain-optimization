from __future__ import annotations

from itertools import product

import numpy as np

from .network import NetworkData, first_stage_cost, recourse_cost
from .network_wasserstein import worst_case_network_recourse


def capacity_grid(data: NetworkData, step: float = 10.0) -> list[np.ndarray]:
    levels = [
        np.arange(0.0, capacity + 0.5 * step, step)
        for capacity in data.supplier_capacity
    ]
    return [np.asarray(values, dtype=float) for values in product(*levels)]


def optimize_nominal_network(
    scenarios: np.ndarray,
    data: NetworkData,
    step: float = 10.0,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for capacity in capacity_grid(data, step):
        recourse = np.mean([recourse_cost(capacity, scenario, data) for scenario in scenarios])
        objective = first_stage_cost(capacity, data) + float(recourse)
        if best is None or objective < best["objective"]:
            best = {
                "capacity": capacity,
                "objective": objective,
                "first_stage_cost": first_stage_cost(capacity, data),
                "expected_recourse": float(recourse),
            }
    if best is None:
        raise RuntimeError("capacity grid is empty")
    return best


def optimize_dro_network(
    scenarios: np.ndarray,
    data: NetworkData,
    epsilon: float,
    step: float = 10.0,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for capacity in capacity_grid(data, step):
        adversary = worst_case_network_recourse(capacity, scenarios, data, epsilon)
        objective = first_stage_cost(capacity, data) + float(adversary["worst_case_recourse"])
        if best is None or objective < best["objective"]:
            best = {
                "capacity": capacity,
                "objective": objective,
                "first_stage_cost": first_stage_cost(capacity, data),
                "worst_case_recourse": float(adversary["worst_case_recourse"]),
                "transport_used": float(adversary["transport_used"]),
            }
    if best is None:
        raise RuntimeError("capacity grid is empty")
    return best


def evaluate_network_policy(
    capacity: np.ndarray,
    scenarios: np.ndarray,
    data: NetworkData,
) -> dict[str, float]:
    recourse = np.asarray([recourse_cost(capacity, scenario, data) for scenario in scenarios])
    total = first_stage_cost(capacity, data) + recourse
    return {
        "mean_total_cost": float(np.mean(total)),
        "p90_total_cost": float(np.quantile(total, 0.90)),
        "worst_total_cost": float(np.max(total)),
        "mean_recourse": float(np.mean(recourse)),
    }
