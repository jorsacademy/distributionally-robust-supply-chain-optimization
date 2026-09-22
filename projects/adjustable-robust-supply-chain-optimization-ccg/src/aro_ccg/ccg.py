from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SupplyChainInstance
from .master import MasterSolution, solve_master
from .recourse import AdversarialResult, OracleMethod, adversarial_oracle
from .uncertainty import UncertaintyScenario, nominal_scenario


@dataclass(frozen=True)
class CCGIteration:
    iteration: int
    lower_bound: float
    upper_bound: float
    current_robust_value: float
    master_eta: float
    worst_recourse: float
    worst_total_shortage: float
    scenario_count: int
    worst_signature: tuple[float, ...]
    oracle_method: str


@dataclass(frozen=True)
class CCGResult:
    robust_objective: float
    lower_bound: float
    upper_bound: float
    open_decision: np.ndarray
    reserved_capacity: np.ndarray
    worst_scenario: UncertaintyScenario
    worst_recourse: float
    worst_shortage: np.ndarray
    iterations: tuple[CCGIteration, ...]
    master_scenario_count: int
    scenarios_scanned_per_oracle: int
    oracle_method: str
    converged: bool


def _first_stage_cost(instance: SupplyChainInstance, master: MasterSolution) -> float:
    return float(
        np.dot(instance.fixed_cost, master.open_decision)
        + np.dot(instance.reservation_cost, master.reserved_capacity)
    )


def solve_ccg(
    instance: SupplyChainInstance,
    *,
    gamma_demand: float,
    gamma_disruption: float,
    tolerance: float = 1e-7,
    max_iterations: int = 50,
    oracle_method: OracleMethod = "auto",
) -> CCGResult:
    """Solve the two-stage robust model by bound-tracked C&CG."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    scenarios: list[UncertaintyScenario] = [nominal_scenario(instance)]
    known_signatures = {scenarios[0].signature}

    best_upper = np.inf
    incumbent_master: MasterSolution | None = None
    incumbent_adversary: AdversarialResult | None = None
    history: list[CCGIteration] = []
    converged = False
    final_lower = -np.inf

    for iteration in range(1, max_iterations + 1):
        master = solve_master(instance, scenarios)
        final_lower = master.objective

        adversary = adversarial_oracle(
            instance,
            master.reserved_capacity,
            gamma_demand=gamma_demand,
            gamma_disruption=gamma_disruption,
            method=oracle_method,
        )
        current_robust = _first_stage_cost(instance, master) + adversary.recourse.objective

        if current_robust < best_upper - 1e-10:
            best_upper = current_robust
            incumbent_master = master
            incumbent_adversary = adversary

        gap = max(0.0, best_upper - final_lower)
        history.append(
            CCGIteration(
                iteration=iteration,
                lower_bound=float(final_lower),
                upper_bound=float(best_upper),
                current_robust_value=float(current_robust),
                master_eta=float(master.eta),
                worst_recourse=float(adversary.recourse.objective),
                worst_total_shortage=float(np.sum(adversary.recourse.shortage)),
                scenario_count=len(scenarios),
                worst_signature=adversary.scenario.signature,
                oracle_method=adversary.oracle_method,
            )
        )

        scale = max(1.0, abs(best_upper))
        if gap <= tolerance * scale:
            converged = True
            break

        signature = adversary.scenario.signature
        if signature in known_signatures:
            raise RuntimeError(
                "C&CG stalled: the worst scenario is already in the master but the bound gap is open"
            )

        scenarios.append(adversary.scenario)
        known_signatures.add(signature)

    if incumbent_master is None or incumbent_adversary is None:
        raise RuntimeError("C&CG did not produce an incumbent solution")

    return CCGResult(
        robust_objective=float(best_upper),
        lower_bound=float(final_lower),
        upper_bound=float(best_upper),
        open_decision=incumbent_master.open_decision.copy(),
        reserved_capacity=incumbent_master.reserved_capacity.copy(),
        worst_scenario=incumbent_adversary.scenario,
        worst_recourse=float(incumbent_adversary.recourse.objective),
        worst_shortage=incumbent_adversary.recourse.shortage.copy(),
        iterations=tuple(history),
        master_scenario_count=len(scenarios),
        scenarios_scanned_per_oracle=incumbent_adversary.scenarios_scanned,
        oracle_method=incumbent_adversary.oracle_method,
        converged=converged,
    )
