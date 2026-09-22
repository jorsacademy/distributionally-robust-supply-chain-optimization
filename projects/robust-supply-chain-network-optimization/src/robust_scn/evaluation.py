from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SupplyChainInstance
from .model import NetworkSolution


@dataclass(frozen=True)
class StressTestResult:
    n_scenarios: int
    scenario_violation_rate: float
    facility_violation_rate: np.ndarray
    mean_excess_load: np.ndarray
    max_excess_load: np.ndarray
    mean_total_excess: float
    p95_total_excess: float


def sample_demand_scenarios(
    instance: SupplyChainInstance,
    n_scenarios: int,
    seed: int,
    shock_probability: float = 0.35,
) -> np.ndarray:
    """
    Generate bounded positive demand shocks for stress testing.

    This is deliberately an evaluation distribution, not a probability model implied by the
    Bertsimas-Sim uncertainty set. Each customer independently receives a shock with the supplied
    probability; shocked magnitudes are beta-distributed fractions of the declared max deviation.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    if not 0.0 <= shock_probability <= 1.0:
        raise ValueError("shock_probability must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    active = rng.random((n_scenarios, instance.n_customers)) < shock_probability
    intensity = rng.beta(2.0, 1.5, size=(n_scenarios, instance.n_customers))
    shocks = active * intensity * instance.demand_deviation[None, :]
    return instance.nominal_demand[None, :] + shocks


def facility_loads(
    solution: NetworkSolution,
    demand_scenarios: np.ndarray,
    n_facilities: int,
) -> np.ndarray:
    demand = np.asarray(demand_scenarios, dtype=float)
    if demand.ndim != 2:
        raise ValueError("demand_scenarios must be two-dimensional")
    if demand.shape[1] != len(solution.assignment):
        raise ValueError("scenario width must equal number of customers")
    loads = np.zeros((demand.shape[0], n_facilities), dtype=float)
    assignment = np.asarray(solution.assignment, dtype=int)
    for i in range(n_facilities):
        loads[:, i] = np.sum(demand[:, assignment == i], axis=1)
    return loads


def stress_test(
    instance: SupplyChainInstance,
    solution: NetworkSolution,
    n_scenarios: int = 5000,
    seed: int = 2026,
    shock_probability: float = 0.35,
) -> StressTestResult:
    scenarios = sample_demand_scenarios(
        instance,
        n_scenarios=n_scenarios,
        seed=seed,
        shock_probability=shock_probability,
    )
    loads = facility_loads(solution, scenarios, instance.n_facilities)
    excess = np.maximum(loads - instance.capacity[None, :], 0.0)
    violated = excess > 1e-9
    total_excess = np.sum(excess, axis=1)
    return StressTestResult(
        n_scenarios=n_scenarios,
        scenario_violation_rate=float(np.mean(np.any(violated, axis=1))),
        facility_violation_rate=np.mean(violated, axis=0),
        mean_excess_load=np.mean(excess, axis=0),
        max_excess_load=np.max(excess, axis=0),
        mean_total_excess=float(np.mean(total_excess)),
        p95_total_excess=float(np.quantile(total_excess, 0.95)),
    )
