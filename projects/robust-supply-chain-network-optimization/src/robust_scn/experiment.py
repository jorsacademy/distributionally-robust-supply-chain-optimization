from __future__ import annotations

import numpy as np

from .data import SupplyChainInstance
from .evaluation import stress_test
from .model import solve_network


def gamma_sweep(
    instance: SupplyChainInstance,
    gammas: list[float] | tuple[float, ...],
    n_scenarios: int = 5000,
    seed: int = 2026,
    shock_probability: float = 0.35,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for gamma in gammas:
        solution = solve_network(instance, gamma=gamma)
        evaluation = stress_test(
            instance,
            solution,
            n_scenarios=n_scenarios,
            seed=seed,
            shock_probability=shock_probability,
        )
        records.append(
            {
                "gamma": float(gamma),
                "objective": solution.objective,
                "open_facilities": [instance.facility_names[i] for i in solution.open_facilities],
                "assignment": {
                    instance.customer_names[j]: instance.facility_names[i]
                    for j, i in enumerate(solution.assignment)
                },
                "minimum_robust_capacity_slack": float(np.min(solution.capacity_slack)),
                "stress_test": {
                    "n_scenarios": evaluation.n_scenarios,
                    "scenario_violation_rate": evaluation.scenario_violation_rate,
                    "mean_total_excess": evaluation.mean_total_excess,
                    "p95_total_excess": evaluation.p95_total_excess,
                    "facility_violation_rate": {
                        instance.facility_names[i]: float(evaluation.facility_violation_rate[i])
                        for i in range(instance.n_facilities)
                    },
                },
            }
        )
    return records
