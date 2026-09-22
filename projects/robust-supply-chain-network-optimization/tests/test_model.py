import numpy as np

from robust_scn import SupplyChainInstance, demo_instance, solve_network


def tiny_instance() -> SupplyChainInstance:
    return SupplyChainInstance(
        facility_names=("A", "B"),
        customer_names=("C1", "C2", "C3"),
        fixed_cost=np.array([10.0, 10.0]),
        capacity=np.array([25.0, 25.0]),
        nominal_demand=np.array([10.0, 10.0, 10.0]),
        demand_deviation=np.array([5.0, 5.0, 5.0]),
        shipping_cost=np.array([[1.0, 1.0, 8.0], [8.0, 1.0, 1.0]]),
    )


def test_nominal_solution_is_feasible_and_single_source() -> None:
    instance = demo_instance()
    solution = solve_network(instance, gamma=0.0)
    assert len(solution.assignment) == instance.n_customers
    assert set(solution.assignment).issubset(set(range(instance.n_facilities)))
    assert np.all(solution.nominal_load <= instance.capacity + 1e-7)
    assert np.all(solution.capacity_slack >= -1e-6)


def test_robust_solution_satisfies_analytical_budgeted_load() -> None:
    instance = demo_instance()
    solution = solve_network(instance, gamma=2.5)
    assert np.all(solution.robust_load <= instance.capacity + 1e-6)
    assert np.all(solution.capacity_slack >= -1e-6)


def test_price_of_robustness_is_nonnegative_on_same_instance() -> None:
    instance = demo_instance()
    nominal = solve_network(instance, gamma=0.0)
    robust = solve_network(instance, gamma=3.0)
    assert robust.objective + 1e-6 >= nominal.objective


def test_tiny_instance_requires_two_facilities() -> None:
    solution = solve_network(tiny_instance(), gamma=1.0)
    assert len(solution.open_facilities) == 2
    assert np.all(solution.robust_load <= 25.0 + 1e-6)
