import numpy as np

from aro_ccg.ccg import solve_ccg
from aro_ccg.data import demo_instance
from aro_ccg.evaluation import audit_solution
from aro_ccg.master import solve_master
from aro_ccg.uncertainty import enumerate_joint_scenarios, nominal_scenario


def test_ccg_with_milp_adversary_matches_full_extreme_scenario_master() -> None:
    instance = demo_instance()
    gamma_demand = 2.0
    gamma_disruption = 1.0

    result = solve_ccg(
        instance,
        gamma_demand=gamma_demand,
        gamma_disruption=gamma_disruption,
        tolerance=1e-8,
        max_iterations=30,
        oracle_method="milp",
    )
    assert result.converged
    assert result.oracle_method == "milp"

    all_scenarios = [nominal_scenario(instance)]
    all_scenarios.extend(
        enumerate_joint_scenarios(
            instance,
            gamma_demand=gamma_demand,
            gamma_disruption=gamma_disruption,
        )
    )
    full = solve_master(instance, all_scenarios)

    np.testing.assert_allclose(
        result.robust_objective,
        full.objective,
        rtol=1e-7,
        atol=1e-7,
    )
    assert result.master_scenario_count < len(all_scenarios)


def test_fractional_budget_uses_enumeration_and_passes_independent_audit() -> None:
    instance = demo_instance()
    result = solve_ccg(
        instance,
        gamma_demand=1.5,
        gamma_disruption=1.0,
        tolerance=1e-8,
        max_iterations=30,
        oracle_method="auto",
    )
    assert result.converged
    assert result.oracle_method == "enumeration"

    audit = audit_solution(
        instance,
        result.open_decision,
        result.reserved_capacity,
        gamma_demand=1.5,
        gamma_disruption=1.0,
        oracle_method="enumeration",
    )

    np.testing.assert_allclose(
        audit.robust_total_cost,
        result.robust_objective,
        rtol=1e-7,
        atol=1e-7,
    )
    assert np.all(result.reserved_capacity >= -1e-9)
    assert np.all(
        result.reserved_capacity
        <= instance.capacity_max * result.open_decision + 1e-7
    )
