from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .ccg import CCGResult, solve_ccg
from .data import SupplyChainInstance
from .recourse import OracleMethod
from .uncertainty import (
    UncertaintyScenario,
    enumerate_joint_scenarios,
    nominal_scenario,
)


@dataclass(frozen=True)
class StaticPlanSolution:
    label: str
    objective: float
    open_decision: np.ndarray
    reserved_capacity: np.ndarray
    shipments: np.ndarray
    shortage: np.ndarray
    scenario_count: int


@dataclass(frozen=True)
class BenchmarkComparison:
    deterministic: StaticPlanSolution
    static_robust: StaticPlanSolution
    adjustable_robust: CCGResult

    @property
    def value_of_adjustability(self) -> float:
        return float(self.static_robust.objective - self.adjustable_robust.robust_objective)

    @property
    def robustness_premium(self) -> float:
        return float(self.adjustable_robust.robust_objective - self.deterministic.objective)


def _solve_static_plan(
    instance: SupplyChainInstance,
    scenarios: list[UncertaintyScenario],
    *,
    label: str,
) -> StaticPlanSolution:
    """
    Solve a non-adaptive plan.

    Activation, capacity, shipment and shortage variables are shared across all scenarios.
    The static robust benchmark therefore differs from adjustable RO only through the
    recourse information structure.
    """
    if not scenarios:
        raise ValueError("static plan requires at least one scenario")

    s_count = instance.n_suppliers
    m_count = instance.n_markets

    open_start = 0
    cap_start = open_start + s_count
    ship_start = cap_start + s_count
    shortage_start = ship_start + s_count * m_count
    n_vars = shortage_start + m_count

    def ship_idx(s: int, m: int) -> int:
        return ship_start + s * m_count + m

    objective = np.zeros(n_vars, dtype=float)
    objective[open_start:cap_start] = instance.fixed_cost
    objective[cap_start:ship_start] = instance.reservation_cost
    objective[ship_start:shortage_start] = instance.shipping_cost.reshape(-1)
    objective[shortage_start:] = instance.shortage_penalty

    integrality = np.zeros(n_vars, dtype=int)
    integrality[open_start:cap_start] = 1

    lower = np.zeros(n_vars, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)
    upper[open_start:cap_start] = 1.0
    upper[cap_start:ship_start] = instance.capacity_max

    n_rows = s_count + len(scenarios) * (s_count + m_count)
    matrix = lil_matrix((n_rows, n_vars), dtype=float)
    lb = np.full(n_rows, -np.inf, dtype=float)
    ub = np.full(n_rows, np.inf, dtype=float)
    row = 0

    for s in range(s_count):
        matrix[row, cap_start + s] = 1.0
        matrix[row, open_start + s] = -float(instance.capacity_max[s])
        ub[row] = 0.0
        row += 1

    for scenario in scenarios:
        for s in range(s_count):
            matrix[row, cap_start + s] = -float(scenario.availability[s])
            for m in range(m_count):
                matrix[row, ship_idx(s, m)] = 1.0
            ub[row] = 0.0
            row += 1

        for m in range(m_count):
            for s in range(s_count):
                matrix[row, ship_idx(s, m)] = -1.0
            matrix[row, shortage_start + m] = -1.0
            ub[row] = -float(scenario.demand[m])
            row += 1

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), lb, ub),
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if result.status != 0 or result.x is None or result.fun is None:
        raise RuntimeError(f"{label} baseline failed: {result.message}")

    vector = np.asarray(result.x, dtype=float)
    return StaticPlanSolution(
        label=label,
        objective=float(result.fun),
        open_decision=vector[open_start:cap_start].copy(),
        reserved_capacity=vector[cap_start:ship_start].copy(),
        shipments=vector[ship_start:shortage_start].reshape(s_count, m_count).copy(),
        shortage=vector[shortage_start:].copy(),
        scenario_count=len(scenarios),
    )


def solve_deterministic_baseline(instance: SupplyChainInstance) -> StaticPlanSolution:
    return _solve_static_plan(
        instance,
        [nominal_scenario(instance)],
        label="deterministic",
    )


def solve_static_robust_baseline(
    instance: SupplyChainInstance,
    *,
    gamma_demand: float,
    gamma_disruption: float,
) -> StaticPlanSolution:
    scenarios = enumerate_joint_scenarios(
        instance,
        gamma_demand=gamma_demand,
        gamma_disruption=gamma_disruption,
    )
    nominal = nominal_scenario(instance)
    unique = {nominal.signature: nominal}
    for scenario in scenarios:
        unique[scenario.signature] = scenario
    return _solve_static_plan(
        instance,
        list(unique.values()),
        label="static_budgeted_robust",
    )


def compare_methods(
    instance: SupplyChainInstance,
    *,
    gamma_demand: float,
    gamma_disruption: float,
    oracle_method: OracleMethod = "auto",
) -> BenchmarkComparison:
    deterministic = solve_deterministic_baseline(instance)
    static_robust = solve_static_robust_baseline(
        instance,
        gamma_demand=gamma_demand,
        gamma_disruption=gamma_disruption,
    )
    adjustable = solve_ccg(
        instance,
        gamma_demand=gamma_demand,
        gamma_disruption=gamma_disruption,
        oracle_method=oracle_method,
    )
    return BenchmarkComparison(
        deterministic=deterministic,
        static_robust=static_robust,
        adjustable_robust=adjustable,
    )
