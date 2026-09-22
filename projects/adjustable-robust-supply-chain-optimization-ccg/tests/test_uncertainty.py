import numpy as np

from aro_ccg.data import demo_instance
from aro_ccg.uncertainty import budget_extreme_points, enumerate_joint_scenarios


def test_zero_budget_is_nominal_factor() -> None:
    points = budget_extreme_points(4, 0.0)
    assert len(points) == 1
    np.testing.assert_allclose(points[0], np.zeros(4))


def test_fractional_budget_extreme_points_are_exact() -> None:
    points = budget_extreme_points(3, 1.5)
    assert len(points) == 6
    for point in points:
        np.testing.assert_allclose(point.sum(), 1.5)
        assert np.count_nonzero(np.isclose(point, 1.0)) == 1
        assert np.count_nonzero(np.isclose(point, 0.5)) == 1
        assert np.all((point >= 0.0) & (point <= 1.0))


def test_joint_scenarios_respect_declared_bounds() -> None:
    instance = demo_instance()
    scenarios = enumerate_joint_scenarios(instance, gamma_demand=2.0, gamma_disruption=1.0)
    assert scenarios

    demand_upper = instance.nominal_demand + instance.demand_deviation
    availability_lower = 1.0 - instance.max_availability_loss

    for scenario in scenarios:
        assert np.all(scenario.demand >= instance.nominal_demand - 1e-10)
        assert np.all(scenario.demand <= demand_upper + 1e-10)
        assert np.all(scenario.availability >= availability_lower - 1e-10)
        assert np.all(scenario.availability <= 1.0 + 1e-10)
        np.testing.assert_allclose(scenario.demand_factors.sum(), 2.0)
        np.testing.assert_allclose(scenario.disruption_factors.sum(), 1.0)
