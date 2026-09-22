from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .data import SupplyChainInstance
from .uncertainty import UncertaintyScenario


@dataclass(frozen=True)
class MasterSolution:
    objective: float
    open_decision: np.ndarray
    reserved_capacity: np.ndarray
    eta: float
    scenario_recourse_costs: np.ndarray
    mip_status: int
    mip_message: str


def solve_master(
    instance: SupplyChainInstance,
    scenarios: list[UncertaintyScenario],
    *,
    time_limit: float | None = 30.0,
    mip_rel_gap: float = 0.0,
) -> MasterSolution:
    """Solve the restricted C&CG master for the supplied uncertainty scenarios."""
    if not scenarios:
        raise ValueError("the master requires at least one scenario")
    if mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be nonnegative")

    s_count = instance.n_suppliers
    m_count = instance.n_markets
    k_count = len(scenarios)

    open_start = 0
    cap_start = open_start + s_count
    eta_idx = cap_start + s_count
    recourse_start = eta_idx + 1

    block_size = s_count * m_count + m_count
    n_vars = recourse_start + k_count * block_size

    def ship_idx(k: int, s: int, m: int) -> int:
        return recourse_start + k * block_size + s * m_count + m

    def shortage_idx(k: int, m: int) -> int:
        return recourse_start + k * block_size + s_count * m_count + m

    objective = np.zeros(n_vars, dtype=float)
    objective[open_start:cap_start] = instance.fixed_cost
    objective[cap_start:eta_idx] = instance.reservation_cost
    objective[eta_idx] = 1.0

    integrality = np.zeros(n_vars, dtype=int)
    integrality[open_start:cap_start] = 1

    lower = np.zeros(n_vars, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)
    upper[open_start:cap_start] = 1.0
    upper[cap_start:eta_idx] = instance.capacity_max

    # Rows:
    # capacity-open linking
    # + per scenario supplier capacity
    # + per scenario market demand
    # + per scenario eta epigraph.
    n_rows = s_count + k_count * (s_count + m_count + 1)
    matrix = lil_matrix((n_rows, n_vars), dtype=float)
    lb = np.full(n_rows, -np.inf, dtype=float)
    ub = np.full(n_rows, np.inf, dtype=float)
    row = 0

    # x_s <= K_s y_s.
    for s in range(s_count):
        matrix[row, cap_start + s] = 1.0
        matrix[row, open_start + s] = -float(instance.capacity_max[s])
        ub[row] = 0.0
        row += 1

    for k, scenario in enumerate(scenarios):
        # sum_m q_sm^k <= availability_s^k x_s.
        for s in range(s_count):
            matrix[row, cap_start + s] = -float(scenario.availability[s])
            for m in range(m_count):
                matrix[row, ship_idx(k, s, m)] = 1.0
            ub[row] = 0.0
            row += 1

        # sum_s q_sm^k + u_m^k >= demand_m^k.
        for m in range(m_count):
            for s in range(s_count):
                matrix[row, ship_idx(k, s, m)] = -1.0
            matrix[row, shortage_idx(k, m)] = -1.0
            ub[row] = -float(scenario.demand[m])
            row += 1

        # recourse_cost_k <= eta.
        matrix[row, eta_idx] = -1.0
        for s in range(s_count):
            for m in range(m_count):
                matrix[row, ship_idx(k, s, m)] = float(instance.shipping_cost[s, m])
        for m in range(m_count):
            matrix[row, shortage_idx(k, m)] = float(instance.shortage_penalty[m])
        ub[row] = 0.0
        row += 1

    assert row == n_rows

    options: dict[str, float | bool] = {
        "disp": False,
        "mip_rel_gap": float(mip_rel_gap),
    }
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive when provided")
        options["time_limit"] = float(time_limit)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), lb, ub),
        options=options,
    )
    if result.status != 0 or result.x is None or result.fun is None:
        raise RuntimeError(f"restricted master failed: {result.message}")

    vector = np.asarray(result.x, dtype=float)
    open_decision = vector[open_start:cap_start]
    reserved_capacity = vector[cap_start:eta_idx]
    eta = float(vector[eta_idx])

    recourse_costs = np.zeros(k_count, dtype=float)
    for k in range(k_count):
        value = 0.0
        for s in range(s_count):
            for m in range(m_count):
                value += instance.shipping_cost[s, m] * vector[ship_idx(k, s, m)]
        for m in range(m_count):
            value += instance.shortage_penalty[m] * vector[shortage_idx(k, m)]
        recourse_costs[k] = value

    return MasterSolution(
        objective=float(result.fun),
        open_decision=open_decision,
        reserved_capacity=reserved_capacity,
        eta=eta,
        scenario_recourse_costs=recourse_costs,
        mip_status=int(result.status),
        mip_message=str(result.message),
    )
