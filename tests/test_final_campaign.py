import numpy as np

from supply_dro.final_campaign import paired_against_nominal, run_final_campaign, summarize
from supply_dro.joint_wasserstein import worst_case_joint_recourse
from supply_dro.network import default_network, first_stage_cost, recourse_cost


def test_joint_wasserstein_zero_radius_matches_empirical_recourse():
    data = default_network()
    capacity = np.array([60.0, 60.0])
    demand = np.array([[45.0, 40.0], [55.0, 48.0], [50.0, 44.0]])
    availability = np.array([[1.0, 1.0], [0.6, 1.0], [1.0, 0.7]])
    result = worst_case_joint_recourse(capacity, demand, availability, data, epsilon=0.0)
    empirical = np.mean(
        [
            recourse_cost(capacity, demand[i], data, availability=availability[i])
            for i in range(len(demand))
        ]
    )
    assert abs(result["worst_case_recourse"] - empirical) < 1e-8


def test_final_campaign_smoke_has_common_paired_blocks():
    result = run_final_campaign(
        train_n=4,
        validation_n=3,
        final_n=4,
        radii=(0.0, 1.0),
        grid_step=40.0,
    )
    rows = result["rows"]
    assert {row.block for row in rows} == {"nominal_final", "demand_shift", "joint_shift"}
    assert {row.policy for row in rows} == {
        "nominal",
        "demand_dro",
        "joint_dro",
        "classical_robust",
    }
    assert len(summarize(rows)) == 12
    comparisons = paired_against_nominal(rows)
    assert len(comparisons) == 9
    assert result["demand_epsilon"] in {0.0, 1.0}
    assert result["joint_epsilon"] in {0.0, 1.0}
    assert all(np.isfinite(row.total_cost) for row in rows)


def test_first_stage_cost_is_included_in_final_objective():
    data = default_network()
    capacity = np.array([40.0, 80.0])
    demand = np.array([60.0, 50.0])
    availability = np.array([0.8, 1.0])
    total = first_stage_cost(capacity, data) + recourse_cost(
        capacity,
        demand,
        data,
        availability=availability,
    )
    assert total > first_stage_cost(capacity, data)
