from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .ccg import solve_ccg
from .data import SupplyChainInstance
from .recourse import OracleMethod


@dataclass(frozen=True)
class SensitivityRecord:
    gamma_demand: float
    gamma_disruption: float
    robust_objective: float
    first_stage_cost: float
    worst_recourse: float
    worst_total_shortage: float
    total_reserved_capacity: float
    open_suppliers: int
    iterations: int
    master_scenarios: int
    final_gap: float
    oracle_method: str


def run_sensitivity(
    instance: SupplyChainInstance,
    *,
    demand_gammas: list[float],
    disruption_gammas: list[float],
    oracle_method: OracleMethod = "auto",
) -> list[SensitivityRecord]:
    records: list[SensitivityRecord] = []
    for gamma_demand, gamma_disruption in product(demand_gammas, disruption_gammas):
        result = solve_ccg(
            instance,
            gamma_demand=float(gamma_demand),
            gamma_disruption=float(gamma_disruption),
            oracle_method=oracle_method,
        )
        first_stage = float(
            np.dot(instance.fixed_cost, result.open_decision)
            + np.dot(instance.reservation_cost, result.reserved_capacity)
        )
        records.append(
            SensitivityRecord(
                gamma_demand=float(gamma_demand),
                gamma_disruption=float(gamma_disruption),
                robust_objective=float(result.robust_objective),
                first_stage_cost=first_stage,
                worst_recourse=float(result.worst_recourse),
                worst_total_shortage=float(np.sum(result.worst_shortage)),
                total_reserved_capacity=float(np.sum(result.reserved_capacity)),
                open_suppliers=int(np.sum(result.open_decision > 0.5)),
                iterations=len(result.iterations),
                master_scenarios=result.master_scenario_count,
                final_gap=float(result.upper_bound - result.lower_bound),
                oracle_method=result.oracle_method,
            )
        )
    return records


def format_sensitivity_table(records: list[SensitivityRecord]) -> str:
    header = (
        "Gamma_d  Gamma_u   objective   capacity   shortage  iter  scen  gap       oracle"
    )
    lines = [header]
    for record in records:
        lines.append(
            f"{record.gamma_demand:>7.2f}"
            f"{record.gamma_disruption:>9.2f}"
            f"{record.robust_objective:>12.3f}"
            f"{record.total_reserved_capacity:>11.3f}"
            f"{record.worst_total_shortage:>11.3f}"
            f"{record.iterations:>6d}"
            f"{record.master_scenarios:>6d}"
            f"{record.final_gap:>10.3g}   "
            f"{record.oracle_method}"
        )
    return "\n".join(lines)
