from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix

from .data import SupplyChainInstance
from .uncertainty import UncertaintyScenario, enumerate_joint_scenarios

OracleMethod = Literal["auto", "milp", "enumeration"]


@dataclass(frozen=True)
class RecourseSolution:
    objective: float
    shipments: np.ndarray
    shortage: np.ndarray


@dataclass(frozen=True)
class AdversarialResult:
    scenario: UncertaintyScenario
    recourse: RecourseSolution
    scenarios_scanned: int
    oracle_method: str
    adversarial_objective: float
    verification_gap: float


def solve_recourse(
    instance: SupplyChainInstance,
    reserved_capacity: np.ndarray,
    scenario: UncertaintyScenario,
) -> RecourseSolution:
    """Solve the continuous second-stage shipment/shortage LP for fixed capacity."""
    capacity = np.asarray(reserved_capacity, dtype=float)
    if capacity.shape != (instance.n_suppliers,):
        raise ValueError("reserved_capacity has the wrong shape")
    if np.any(capacity < -1e-10):
        raise ValueError("reserved_capacity must be nonnegative")

    s_count = instance.n_suppliers
    m_count = instance.n_markets
    n_ship = s_count * m_count
    n_vars = n_ship + m_count

    def ship_idx(s: int, m: int) -> int:
        return s * m_count + m

    objective = np.zeros(n_vars, dtype=float)
    objective[:n_ship] = instance.shipping_cost.reshape(-1)
    objective[n_ship:] = instance.shortage_penalty

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for s in range(s_count):
        row = np.zeros(n_vars, dtype=float)
        for m in range(m_count):
            row[ship_idx(s, m)] = 1.0
        rows.append(row)
        rhs.append(float(scenario.availability[s] * capacity[s]))

    for m in range(m_count):
        row = np.zeros(n_vars, dtype=float)
        for s in range(s_count):
            row[ship_idx(s, m)] = -1.0
        row[n_ship + m] = -1.0
        rows.append(row)
        rhs.append(float(-scenario.demand[m]))

    result = linprog(
        c=objective,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"recourse LP failed: {result.message}")

    shipments = np.asarray(result.x[:n_ship], dtype=float).reshape(s_count, m_count)
    shortage = np.asarray(result.x[n_ship:], dtype=float)
    return RecourseSolution(
        objective=float(result.fun),
        shipments=shipments,
        shortage=shortage,
    )


def _enumeration_adversary(
    instance: SupplyChainInstance,
    reserved_capacity: np.ndarray,
    gamma_demand: float,
    gamma_disruption: float,
) -> AdversarialResult:
    scenarios = enumerate_joint_scenarios(
        instance,
        gamma_demand=gamma_demand,
        gamma_disruption=gamma_disruption,
    )
    if not scenarios:
        raise RuntimeError("uncertainty oracle generated no scenarios")

    worst_scenario = scenarios[0]
    worst_recourse = solve_recourse(instance, reserved_capacity, worst_scenario)

    for scenario in scenarios[1:]:
        recourse = solve_recourse(instance, reserved_capacity, scenario)
        if recourse.objective > worst_recourse.objective + 1e-10:
            worst_scenario = scenario
            worst_recourse = recourse

    return AdversarialResult(
        scenario=worst_scenario,
        recourse=worst_recourse,
        scenarios_scanned=len(scenarios),
        oracle_method="enumeration",
        adversarial_objective=float(worst_recourse.objective),
        verification_gap=0.0,
    )


def _as_integer_budget(gamma: float, dimension: int) -> int:
    if gamma < 0:
        raise ValueError("uncertainty budgets must be nonnegative")
    capped = min(float(gamma), float(dimension))
    rounded = round(capped)
    if abs(capped - rounded) > 1e-9:
        raise ValueError(
            "the dualized MILP adversary supports integer uncertainty budgets; "
            "use enumeration or method='auto' for fractional budgets"
        )
    return rounded


def _dual_alpha_upper_bounds(instance: SupplyChainInstance) -> np.ndarray:
    """
    Return safe bounds for supplier-capacity dual variables.

    In the recourse dual, alpha_s only relaxes beta_m <= shipping_cost_sm + alpha_s.
    Since beta_m <= shortage_penalty_m, alpha_s above
    max_m(shortage_penalty_m - shipping_cost_sm, 0) cannot improve the dual value.
    """
    return np.maximum(
        np.max(
            instance.shortage_penalty[None, :] - instance.shipping_cost,
            axis=1,
        ),
        0.0,
    )


def dualized_adversarial_oracle(
    instance: SupplyChainInstance,
    reserved_capacity: np.ndarray,
    gamma_demand: float,
    gamma_disruption: float,
) -> AdversarialResult:
    """
    Solve the adversarial subproblem as a MILP for integer uncertainty budgets.

    The continuous recourse LP is dualized. Binary demand/disruption factors multiply
    bounded dual variables; standard exact binary-continuous linearizations produce
    a MILP. The returned scenario is then re-solved through the primal recourse LP,
    and primal/dual agreement is checked explicitly.
    """
    capacity = np.asarray(reserved_capacity, dtype=float)
    if capacity.shape != (instance.n_suppliers,):
        raise ValueError("reserved_capacity has the wrong shape")
    if np.any(capacity < -1e-10):
        raise ValueError("reserved_capacity must be nonnegative")

    s_count = instance.n_suppliers
    m_count = instance.n_markets
    demand_budget = _as_integer_budget(gamma_demand, m_count)
    disruption_budget = _as_integer_budget(gamma_disruption, s_count)
    alpha_ub = _dual_alpha_upper_bounds(instance)

    alpha_start = 0
    beta_start = alpha_start + s_count
    z_start = beta_start + m_count
    w_start = z_start + m_count
    t_start = w_start + s_count
    r_start = t_start + m_count
    n_vars = r_start + s_count

    objective = np.zeros(n_vars, dtype=float)
    objective[alpha_start:beta_start] = capacity
    objective[beta_start:z_start] = -instance.nominal_demand
    objective[t_start:r_start] = -instance.demand_deviation
    objective[r_start:] = -(capacity * instance.max_availability_loss)

    integrality = np.zeros(n_vars, dtype=int)
    integrality[z_start:w_start] = 1
    integrality[w_start:t_start] = 1

    lower = np.zeros(n_vars, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)
    upper[alpha_start:beta_start] = alpha_ub
    upper[beta_start:z_start] = instance.shortage_penalty
    upper[z_start:w_start] = 1.0
    upper[w_start:t_start] = 1.0
    upper[t_start:r_start] = instance.shortage_penalty
    upper[r_start:] = alpha_ub

    rows: list[tuple[dict[int, float], float]] = []

    # Recourse-dual feasibility: beta_m - alpha_s <= shipping_cost_sm.
    for s in range(s_count):
        for m in range(m_count):
            rows.append(
                (
                    {
                        beta_start + m: 1.0,
                        alpha_start + s: -1.0,
                    },
                    float(instance.shipping_cost[s, m]),
                )
            )

    # Budget constraints.
    rows.append(({z_start + m: 1.0 for m in range(m_count)}, float(demand_budget)))
    rows.append(({w_start + s: 1.0 for s in range(s_count)}, float(disruption_budget)))

    # t_m = z_m * beta_m.
    for m in range(m_count):
        p = float(instance.shortage_penalty[m])
        rows.append(({t_start + m: 1.0, beta_start + m: -1.0}, 0.0))
        rows.append(({t_start + m: 1.0, z_start + m: -p}, 0.0))
        rows.append(
            (
                {
                    beta_start + m: 1.0,
                    t_start + m: -1.0,
                    z_start + m: p,
                },
                p,
            )
        )

    # r_s = w_s * alpha_s.
    for s in range(s_count):
        big_m = float(alpha_ub[s])
        rows.append(({r_start + s: 1.0, alpha_start + s: -1.0}, 0.0))
        rows.append(({r_start + s: 1.0, w_start + s: -big_m}, 0.0))
        rows.append(
            (
                {
                    alpha_start + s: 1.0,
                    r_start + s: -1.0,
                    w_start + s: big_m,
                },
                big_m,
            )
        )

    matrix = lil_matrix((len(rows), n_vars), dtype=float)
    ub = np.zeros(len(rows), dtype=float)
    for i, (coefficients, rhs) in enumerate(rows):
        for index, coefficient in coefficients.items():
            matrix[i, index] = coefficient
        ub[i] = rhs

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix.tocsr(),
            np.full(len(rows), -np.inf, dtype=float),
            ub,
        ),
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if result.status != 0 or result.x is None or result.fun is None:
        raise RuntimeError(f"dualized adversarial MILP failed: {result.message}")

    vector = np.asarray(result.x, dtype=float)
    z = np.rint(vector[z_start:w_start]).astype(float)
    w = np.rint(vector[w_start:t_start]).astype(float)
    scenario = UncertaintyScenario(
        demand=instance.nominal_demand + instance.demand_deviation * z,
        availability=1.0 - instance.max_availability_loss * w,
        demand_factors=z,
        disruption_factors=w,
    )

    recourse = solve_recourse(instance, capacity, scenario)
    adversarial_objective = float(-result.fun)
    verification_gap = abs(adversarial_objective - recourse.objective)
    scale = max(1.0, abs(adversarial_objective), abs(recourse.objective))
    if verification_gap > 1e-6 * scale:
        raise RuntimeError(
            "adversarial MILP failed primal/dual verification: "
            f"dual={adversarial_objective}, primal={recourse.objective}"
        )

    return AdversarialResult(
        scenario=scenario,
        recourse=recourse,
        scenarios_scanned=0,
        oracle_method="milp",
        adversarial_objective=adversarial_objective,
        verification_gap=float(verification_gap),
    )


def adversarial_oracle(
    instance: SupplyChainInstance,
    reserved_capacity: np.ndarray,
    gamma_demand: float,
    gamma_disruption: float,
    *,
    method: OracleMethod = "auto",
) -> AdversarialResult:
    """
    Return an exact worst-case recourse scenario.

    auto uses the dualized MILP for integer budgets and the exact extreme-point
    enumeration oracle for fractional budgets.
    """
    if method not in {"auto", "milp", "enumeration"}:
        raise ValueError("unknown adversarial oracle method")

    if method == "enumeration":
        return _enumeration_adversary(
            instance,
            reserved_capacity,
            gamma_demand,
            gamma_disruption,
        )

    if method == "milp":
        return dualized_adversarial_oracle(
            instance,
            reserved_capacity,
            gamma_demand,
            gamma_disruption,
        )

    demand_capped = min(float(gamma_demand), float(instance.n_markets))
    disruption_capped = min(float(gamma_disruption), float(instance.n_suppliers))
    integer_budgets = (
        abs(demand_capped - round(demand_capped)) <= 1e-9
        and abs(disruption_capped - round(disruption_capped)) <= 1e-9
    )
    if integer_budgets:
        return dualized_adversarial_oracle(
            instance,
            reserved_capacity,
            gamma_demand,
            gamma_disruption,
        )
    return _enumeration_adversary(
        instance,
        reserved_capacity,
        gamma_demand,
        gamma_disruption,
    )
