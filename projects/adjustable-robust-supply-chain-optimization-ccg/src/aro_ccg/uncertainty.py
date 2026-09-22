from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .data import SupplyChainInstance


@dataclass(frozen=True)
class UncertaintyScenario:
    demand: np.ndarray
    availability: np.ndarray
    demand_factors: np.ndarray
    disruption_factors: np.ndarray

    @property
    def signature(self) -> tuple[float, ...]:
        values = np.concatenate((self.demand, self.availability))
        return tuple(float(v) for v in np.round(values, 10))


def budget_extreme_points(dimension: int, gamma: float) -> list[np.ndarray]:
    """
    Enumerate the full-budget extreme points needed by the monotone adversarial oracle.

    For U = {z in [0,1]^n : sum(z) <= Gamma}, recourse cost in this benchmark is
    monotone in each uncertainty factor. Therefore a worst case can be chosen from
    the full-budget frontier sum(z) = min(Gamma, n).

    Integer Gamma gives binary extreme points with exactly Gamma active components.
    Fractional Gamma gives floor(Gamma) ones and one component equal to the fraction.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")

    capped = min(float(gamma), float(dimension))
    if capped <= 1e-12:
        return [np.zeros(dimension, dtype=float)]
    if capped >= dimension - 1e-12:
        return [np.ones(dimension, dtype=float)]

    whole = int(np.floor(capped + 1e-12))
    fraction = capped - whole
    points: list[np.ndarray] = []

    if fraction <= 1e-12:
        for active in combinations(range(dimension), whole):
            z = np.zeros(dimension, dtype=float)
            z[list(active)] = 1.0
            points.append(z)
    else:
        for active in combinations(range(dimension), whole):
            remaining = [i for i in range(dimension) if i not in active]
            for fractional_index in remaining:
                z = np.zeros(dimension, dtype=float)
                if active:
                    z[list(active)] = 1.0
                z[fractional_index] = fraction
                points.append(z)

    return points


def nominal_scenario(instance: SupplyChainInstance) -> UncertaintyScenario:
    return UncertaintyScenario(
        demand=instance.nominal_demand.copy(),
        availability=np.ones(instance.n_suppliers, dtype=float),
        demand_factors=np.zeros(instance.n_markets, dtype=float),
        disruption_factors=np.zeros(instance.n_suppliers, dtype=float),
    )


def enumerate_joint_scenarios(
    instance: SupplyChainInstance,
    gamma_demand: float,
    gamma_disruption: float,
) -> list[UncertaintyScenario]:
    demand_points = budget_extreme_points(instance.n_markets, gamma_demand)
    disruption_points = budget_extreme_points(instance.n_suppliers, gamma_disruption)

    scenarios: list[UncertaintyScenario] = []
    for z, w in product(demand_points, disruption_points):
        demand = instance.nominal_demand + instance.demand_deviation * z
        availability = 1.0 - instance.max_availability_loss * w
        scenarios.append(
            UncertaintyScenario(
                demand=demand,
                availability=availability,
                demand_factors=z.copy(),
                disruption_factors=w.copy(),
            )
        )

    unique: dict[tuple[float, ...], UncertaintyScenario] = {}
    for scenario in scenarios:
        unique[scenario.signature] = scenario
    return list(unique.values())
