import numpy as np
import pytest

from aro_ccg.data import demo_instance
from aro_ccg.recourse import adversarial_oracle, dualized_adversarial_oracle


def test_dualized_milp_matches_exact_enumeration() -> None:
    instance = demo_instance()
    capacity = np.array([72.0, 90.0, 82.0])

    milp_result = dualized_adversarial_oracle(
        instance,
        capacity,
        gamma_demand=2.0,
        gamma_disruption=1.0,
    )
    enumeration_result = adversarial_oracle(
        instance,
        capacity,
        gamma_demand=2.0,
        gamma_disruption=1.0,
        method="enumeration",
    )

    np.testing.assert_allclose(
        milp_result.recourse.objective,
        enumeration_result.recourse.objective,
        rtol=1e-8,
        atol=1e-8,
    )
    assert milp_result.verification_gap <= 1e-7


def test_dualized_milp_rejects_fractional_budget() -> None:
    instance = demo_instance()
    with pytest.raises(ValueError, match="integer uncertainty budgets"):
        dualized_adversarial_oracle(
            instance,
            np.array([70.0, 80.0, 75.0]),
            gamma_demand=1.5,
            gamma_disruption=1.0,
        )


def test_auto_uses_enumeration_for_fractional_budget() -> None:
    instance = demo_instance()
    result = adversarial_oracle(
        instance,
        np.array([70.0, 80.0, 75.0]),
        gamma_demand=1.5,
        gamma_disruption=1.0,
        method="auto",
    )
    assert result.oracle_method == "enumeration"
    assert result.scenarios_scanned > 0
