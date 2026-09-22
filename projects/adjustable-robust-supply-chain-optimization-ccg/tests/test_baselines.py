import numpy as np

from aro_ccg.baselines import compare_methods
from aro_ccg.data import demo_instance


def test_policy_class_ordering_on_demo_instance() -> None:
    instance = demo_instance()
    comparison = compare_methods(
        instance,
        gamma_demand=2.0,
        gamma_disruption=1.0,
        oracle_method="milp",
    )

    assert comparison.deterministic.objective <= comparison.adjustable_robust.robust_objective + 1e-7
    assert comparison.adjustable_robust.robust_objective <= comparison.static_robust.objective + 1e-7
    assert comparison.value_of_adjustability >= -1e-7
    assert comparison.robustness_premium >= -1e-7

    assert np.all(
        comparison.adjustable_robust.reserved_capacity
        <= instance.capacity_max * comparison.adjustable_robust.open_decision + 1e-7
    )
