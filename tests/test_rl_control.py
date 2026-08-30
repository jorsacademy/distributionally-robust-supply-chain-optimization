import numpy as np

from supply_dro.rl_control import (
    DynamicSupplyControl,
    q_learning_action,
    solve_dynamic_program,
    train_q_learning,
)
from supply_dro.rl_experiment import run_control_benchmark


def test_dynamic_control_probabilities_and_actions_are_valid():
    env = DynamicSupplyControl()
    assert np.allclose(env.signal_transition.sum(axis=1), 1.0)
    midpoint = env.capacity_index(np.array([40.0, 40.0]))
    feasible = env.feasible_actions(midpoint)
    assert len(feasible) > 1
    for action in feasible:
        assert np.all(
            np.abs(env.capacity_actions[action] - env.capacity_actions[midpoint])
            <= env.config.max_adjustment + 1e-12
        )


def test_exact_dp_and_q_learning_return_feasible_controls():
    env = DynamicSupplyControl()
    values, policy = solve_dynamic_program(env)
    assert np.all(np.isfinite(values))
    learned = train_q_learning(env, seed=3, episodes=250)
    midpoint = env.capacity_index(np.array([40.0, 40.0]))
    for signal in (0, 1):
        exact_action = int(policy[0, midpoint, signal])
        learned_action = q_learning_action(env, learned, 0, midpoint, signal)
        assert exact_action in env.feasible_actions(midpoint)
        assert learned_action in env.feasible_actions(midpoint)


def test_control_benchmark_keeps_model_seed_aggregation_and_common_environment_seeds():
    result = run_control_benchmark(
        environment_seeds=(100, 101),
        model_seeds=(0, 1),
        episodes=120,
    )
    methods = {row["method"] for row in result["summaries"]}
    assert methods == {
        "exact_dynamic_dp",
        "myopic_or",
        "q_learning",
        "static_nominal",
        "static_robust",
    }
    q_rows = [row for row in result["rows"] if row.method == "q_learning"]
    assert len(q_rows) == 2
    assert all(row.model_seed is None for row in q_rows)
    assert {row.environment_seed for row in q_rows} == {100, 101}
