from __future__ import annotations

import numpy as np

from .joint_wasserstein import worst_case_joint_recourse
from .network import NetworkData, first_stage_cost, recourse_cost
from .network_optimize import capacity_grid, optimize_dro_network


def optimize_joint_dro_network(
    demand_scenarios: np.ndarray,
    availability_scenarios: np.ndarray,
    data: NetworkData,
    epsilon: float,
    *,
    step: float = 10.0,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for capacity in capacity_grid(data, step):
        adversary = worst_case_joint_recourse(
            capacity,
            demand_scenarios,
            availability_scenarios,
            data,
            epsilon,
        )
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


def validation_tail_cost(
    capacity: np.ndarray,
    demand: np.ndarray,
    availability: np.ndarray,
    data: NetworkData,
    quantile: float = 0.90,
) -> float:
    costs = np.asarray(
        [
            first_stage_cost(capacity, data)
            + recourse_cost(capacity, demand[i], data, availability=availability[i])
            for i in range(len(demand))
        ]
    )
    return float(np.quantile(costs, quantile))


def calibrate_demand_radius(
    train_demand: np.ndarray,
    validation_demand: np.ndarray,
    data: NetworkData,
    radii: tuple[float, ...],
    *,
    step: float = 10.0,
) -> dict[str, object]:
    validation_availability = np.ones((len(validation_demand), len(data.supplier_capacity)))
    candidates = []
    for epsilon in radii:
        decision = optimize_dro_network(train_demand, data, epsilon, step=step)
        score = validation_tail_cost(
            decision["capacity"], validation_demand, validation_availability, data
        )
        candidates.append((score, epsilon, decision))
    score, epsilon, decision = min(candidates, key=lambda item: (item[0], item[1]))
    return {"epsilon": float(epsilon), "validation_p90": float(score), "decision": decision}


def calibrate_joint_radius(
    train_demand: np.ndarray,
    train_availability: np.ndarray,
    validation_demand: np.ndarray,
    validation_availability: np.ndarray,
    data: NetworkData,
    radii: tuple[float, ...],
    *,
    step: float = 10.0,
) -> dict[str, object]:
    candidates = []
    for epsilon in radii:
        decision = optimize_joint_dro_network(
            train_demand,
            train_availability,
            data,
            epsilon,
            step=step,
        )
        score = validation_tail_cost(
            decision["capacity"], validation_demand, validation_availability, data
        )
        candidates.append((score, epsilon, decision))
    score, epsilon, decision = min(candidates, key=lambda item: (item[0], item[1]))
    return {"epsilon": float(epsilon), "validation_p90": float(score), "decision": decision}
