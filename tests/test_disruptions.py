import numpy as np

from supply_dro.network import default_network, disruption_scenarios, recourse_cost
from supply_dro.network_optimize import (
    evaluate_disruption_policy,
    optimize_classical_robust_network,
)


def test_supplier_disruption_cannot_reduce_recourse_cost():
    data = default_network()
    capacity = np.array([60.0, 60.0])
    demand = np.array([50.0, 45.0])
    nominal = recourse_cost(capacity, demand, data)
    disrupted = recourse_cost(capacity, demand, data, availability=np.array([0.35, 1.0]))
    assert disrupted + 1e-9 >= nominal


def test_classical_robust_policy_is_feasible_and_evaluable():
    data = default_network()
    stress = disruption_scenarios()
    robust = optimize_classical_robust_network(stress, data, step=20.0)
    capacity = robust["capacity"]
    assert np.all(capacity >= 0.0)
    assert np.all(capacity <= data.supplier_capacity + 1e-9)
    evaluation = evaluate_disruption_policy(capacity, stress, data)
    assert evaluation["worst_stress_cost"] >= evaluation["mean_stress_cost"]
    assert evaluation["worst_stress_scenario"] in {name for _, _, name in stress}
