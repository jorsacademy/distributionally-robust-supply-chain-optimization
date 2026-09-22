import numpy as np

from aro_ccg.data import demo_instance
from aro_ccg.recourse import solve_recourse
from aro_ccg.uncertainty import nominal_scenario


def test_zero_capacity_is_served_by_explicit_shortage() -> None:
    instance = demo_instance()
    scenario = nominal_scenario(instance)
    result = solve_recourse(
        instance,
        reserved_capacity=np.zeros(instance.n_suppliers),
        scenario=scenario,
    )

    np.testing.assert_allclose(result.shipments, 0.0, atol=1e-9)
    np.testing.assert_allclose(result.shortage, scenario.demand, atol=1e-9)
    expected = float(np.dot(instance.shortage_penalty, scenario.demand))
    np.testing.assert_allclose(result.objective, expected, rtol=1e-9, atol=1e-9)


def test_more_capacity_cannot_increase_optimal_recourse_cost() -> None:
    instance = demo_instance()
    scenario = nominal_scenario(instance)

    low = solve_recourse(
        instance,
        reserved_capacity=np.array([20.0, 20.0, 20.0]),
        scenario=scenario,
    )
    high = solve_recourse(
        instance,
        reserved_capacity=np.array([80.0, 80.0, 80.0]),
        scenario=scenario,
    )

    assert high.objective <= low.objective + 1e-8
