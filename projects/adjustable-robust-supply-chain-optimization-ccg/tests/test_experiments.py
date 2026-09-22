from aro_ccg.data import demo_instance
from aro_ccg.experiments import run_sensitivity


def test_sensitivity_reports_expected_diagnostics() -> None:
    records = run_sensitivity(
        demo_instance(),
        demand_gammas=[0.0, 1.0, 2.0],
        disruption_gammas=[1.0],
        oracle_method="milp",
    )

    assert len(records) == 3
    objectives = [record.robust_objective for record in records]
    assert objectives == sorted(objectives)

    for record in records:
        assert record.iterations >= 1
        assert record.master_scenarios >= 1
        assert record.final_gap >= -1e-7
        assert record.worst_total_shortage >= -1e-9
        assert record.oracle_method == "milp"
