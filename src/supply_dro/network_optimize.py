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


def optimize_classical_robust_network(
    stress_scenarios: list[tuple[np.ndarray, np.ndarray, str]],
    data: NetworkData,
    step: float = 10.0,
) -> dict[str, object]:
    """Minimize worst-case total cost over a finite demand/disruption stress set."""
    best: dict[str, object] | None = None
    for capacity in capacity_grid(data, step):
        scenario_costs = []
        for demand, availability, name in stress_scenarios:
            total = first_stage_cost(capacity, data) + recourse_cost(
                capacity, demand, data, availability=availability
            )
            scenario_costs.append((float(total), name))
        worst_cost, worst_name = max(scenario_costs, key=lambda item: item[0])
        if best is None or worst_cost < best["objective"]:
            best = {
                "capacity": capacity,
                "objective": worst_cost,
                "worst_scenario": worst_name,
                "first_stage_cost": first_stage_cost(capacity, data),
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


def evaluate_disruption_policy(
    capacity: np.ndarray,
    stress_scenarios: list[tuple[np.ndarray, np.ndarray, str]],
    data: NetworkData,
) -> dict[str, float | str]:
    costs = []
    for demand, availability, name in stress_scenarios:
        total = first_stage_cost(capacity, data) + recourse_cost(
            capacity, demand, data, availability=availability
        )
        costs.append((float(total), name))
    worst_cost, worst_name = max(costs, key=lambda item: item[0])
    return {
        "mean_stress_cost": float(np.mean([cost for cost, _ in costs])),
        "worst_stress_cost": worst_cost,
        "worst_stress_scenario": worst_name,
    }
