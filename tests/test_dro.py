from supply_dro.model import CostModel, empirical_demand
from supply_dro.optimize import optimize_dro, optimize_nominal
from supply_dro.wasserstein import worst_case_expected_cost


def test_zero_radius_matches_empirical_expectation():
    model = CostModel()
    demand = empirical_demand(n=12)
    q = 50.0
    inner = worst_case_expected_cost(q, demand, model, epsilon=0.0)
    empirical_mean = float(model.cost(q, demand).mean())
    assert abs(inner["worst_case_cost"] - empirical_mean) < 1e-7


def test_worst_case_cost_is_monotone_in_radius():
    model = CostModel()
    demand = empirical_demand(n=12)
    c0 = worst_case_expected_cost(50.0, demand, model, epsilon=0.0)["worst_case_cost"]
    c5 = worst_case_expected_cost(50.0, demand, model, epsilon=5.0)["worst_case_cost"]
    assert c5 >= c0 - 1e-8


def test_optimizers_return_grid_feasible_quantities():
    model = CostModel()
    demand = empirical_demand(n=15)
    nominal = optimize_nominal(demand, model)
    robust = optimize_dro(demand, model, epsilon=3.0)
    assert nominal["order_quantity"] >= 0
    assert robust["order_quantity"] >= 0
    assert robust["objective"] >= 0
