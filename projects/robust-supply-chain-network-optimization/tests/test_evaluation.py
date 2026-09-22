import numpy as np

from robust_scn import demo_instance, solve_network, stress_test
from robust_scn.evaluation import facility_loads, sample_demand_scenarios


def test_scenario_generation_is_reproducible_and_bounded() -> None:
    instance = demo_instance()
    first = sample_demand_scenarios(instance, 50, seed=7)
    second = sample_demand_scenarios(instance, 50, seed=7)
    assert np.array_equal(first, second)
    assert np.all(first >= instance.nominal_demand[None, :])
    assert np.all(
        first <= instance.nominal_demand[None, :] + instance.demand_deviation[None, :] + 1e-12
    )


def test_facility_loads_conserve_total_demand() -> None:
    instance = demo_instance()
    solution = solve_network(instance, gamma=1.0)
    scenarios = sample_demand_scenarios(instance, 20, seed=11)
    loads = facility_loads(solution, scenarios, instance.n_facilities)
    assert np.allclose(np.sum(loads, axis=1), np.sum(scenarios, axis=1))


def test_stress_test_rates_are_valid_probabilities() -> None:
    instance = demo_instance()
    solution = solve_network(instance, gamma=2.0)
    result = stress_test(instance, solution, n_scenarios=100, seed=5)
    assert 0.0 <= result.scenario_violation_rate <= 1.0
    assert np.all((result.facility_violation_rate >= 0.0) & (result.facility_violation_rate <= 1.0))
    assert result.mean_total_excess >= 0.0
    assert result.p95_total_excess >= 0.0
