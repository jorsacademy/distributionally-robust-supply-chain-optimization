from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SupplyChainInstance
from .recourse import OracleMethod, adversarial_oracle


@dataclass(frozen=True)
class RobustAudit:
    first_stage_cost: float
    worst_recourse: float
    robust_total_cost: float
    worst_demand: np.ndarray
    worst_availability: np.ndarray
    worst_shortage: np.ndarray
    scenarios_scanned: int
    oracle_method: str


def audit_solution(
    instance: SupplyChainInstance,
    open_decision: np.ndarray,
    reserved_capacity: np.ndarray,
    *,
    gamma_demand: float,
    gamma_disruption: float,
    oracle_method: OracleMethod = "enumeration",
) -> RobustAudit:
    """Independently evaluate a fixed first-stage decision against the uncertainty set."""
    open_values = np.asarray(open_decision, dtype=float)
    capacity = np.asarray(reserved_capacity, dtype=float)
    if open_values.shape != (instance.n_suppliers,):
        raise ValueError("open_decision has the wrong shape")
    if capacity.shape != (instance.n_suppliers,):
        raise ValueError("reserved_capacity has the wrong shape")
    if np.any(capacity > instance.capacity_max * open_values + 1e-7):
        raise ValueError("capacity violates activation linking")

    first_stage = float(
        np.dot(instance.fixed_cost, open_values)
        + np.dot(instance.reservation_cost, capacity)
    )
    adversary = adversarial_oracle(
        instance,
        capacity,
        gamma_demand=gamma_demand,
        gamma_disruption=gamma_disruption,
        method=oracle_method,
    )
    return RobustAudit(
        first_stage_cost=first_stage,
        worst_recourse=float(adversary.recourse.objective),
        robust_total_cost=first_stage + float(adversary.recourse.objective),
        worst_demand=adversary.scenario.demand.copy(),
        worst_availability=adversary.scenario.availability.copy(),
        worst_shortage=adversary.recourse.shortage.copy(),
        scenarios_scanned=adversary.scenarios_scanned,
        oracle_method=adversary.oracle_method,
    )
