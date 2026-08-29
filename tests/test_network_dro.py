import numpy as np

from supply_dro.network import default_network, demand_scenarios, recourse_cost
from supply_dro.network_optimize import optimize_dro_network, optimize_nominal_network
from supply_dro.network_wasserstein import worst_case_network_recourse


def test_zero_radius_matches_empirical_mean_recourse():
    data = default_network()
    scenarios = demand_scenarios(seed=3, n=6)
    capacity = np.array([50.0, 50.0])
    empirical = np.mean([recourse_cost(capacity, scenario, data) for scenario in scenarios])
    worst = worst_case_network_recourse(capacity, scenarios, data, epsilon=0.0)
    assert abs(empirical - worst["worst_case_recourse"]) < 1e-7


def test_wasserstein_cost_is_monotone_in_radius():
    data = default_network()
    scenarios = demand_scenarios(seed=4, n=6)
    capacity = np.array([40.0, 50.0])
    low = worst_case_network_recourse(capacity, scenarios, data, epsilon=0.0)
    high = worst_case_network_recourse(capacity, scenarios, data, epsilon=10.0)
    assert high["worst_case_recourse"] >= low["worst_case_recourse"] - 1e-8


def test_nominal_and_dro_optimizers_return_feasible_capacities():
    data = default_network()
    scenarios = demand_scenarios(seed=5, n=6)
    nominal = optimize_nominal_network(scenarios, data, step=20.0)
    robust = optimize_dro_network(scenarios, data, epsilon=8.0, step=20.0)
    for result in (nominal, robust):
        capacity = result["capacity"]
        assert np.all(capacity >= 0.0)
        assert np.all(capacity <= data.supplier_capacity + 1e-8)
