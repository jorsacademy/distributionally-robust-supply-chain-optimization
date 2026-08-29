from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .network import NetworkData, recourse_cost


def worst_case_network_recourse(
    reserved_capacity: np.ndarray,
    scenarios: np.ndarray,
    data: NetworkData,
    epsilon: float,
) -> dict[str, object]:
    """Finite-support 1-Wasserstein adversary over vector demand scenarios."""
    scenarios = np.asarray(scenarios, dtype=float)
    n = scenarios.shape[0]
    source_prob = np.full(n, 1.0 / n)
    scenario_cost = np.asarray(
        [recourse_cost(reserved_capacity, scenario, data) for scenario in scenarios]
    )

    objective = -np.tile(scenario_cost, n)
    a_eq = np.zeros((n, n * n))
    for source in range(n):
        a_eq[source, source * n:(source + 1) * n] = 1.0

    distance = np.abs(scenarios[:, None, :] - scenarios[None, :, :]).sum(axis=2)
    result = linprog(
        objective,
        A_ub=distance.reshape(1, -1),
        b_ub=np.array([float(epsilon)]),
        A_eq=a_eq,
        b_eq=source_prob,
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
        "scenario_cost": scenario_cost,
    }
