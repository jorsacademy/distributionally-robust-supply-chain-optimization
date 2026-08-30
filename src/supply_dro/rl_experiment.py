from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .network import default_network, demand_scenarios, disruption_scenarios
from .network_optimize import optimize_classical_robust_network, optimize_nominal_network
from .rl_control import (
    DynamicSupplyControl,
    q_learning_action,
    solve_dynamic_program,
    train_q_learning,
)
from .statistics import paired_bootstrap


@dataclass(frozen=True)
class ControlEpisode:
    method: str
    environment_seed: int
    model_seed: int | None
    total_cost: float
    mean_decision_latency_ms: float


def _step_toward_target(env: DynamicSupplyControl, current: int, target: np.ndarray) -> int:
    feasible = env.feasible_actions(current)
    distances = [
        float(np.sum(np.abs(env.capacity_actions[action] - target)))
        for action in feasible
    ]
    return int(feasible[int(np.argmin(distances))])


def _rollout(
    env: DynamicSupplyControl,
    *,
    method: str,
    environment_seed: int,
    policy_fn,
    model_seed: int | None = None,
) -> ControlEpisode:
    rng = np.random.default_rng(environment_seed)
    current = env.capacity_index(np.array([40.0, 40.0]))
    signal = 0
    total_cost = 0.0
    latencies = []
    for t in range(env.config.horizon):
        start = perf_counter()
        action = int(policy_fn(t, current, signal))
        latencies.append((perf_counter() - start) * 1000.0)
        cost, next_signal = env.sample_step(
            signal=signal,
            current_index=current,
            action_index=action,
            rng=rng,
        )
        total_cost += cost
        current = action
        signal = next_signal
    return ControlEpisode(
        method=method,
        environment_seed=environment_seed,
        model_seed=model_seed,
        total_cost=float(total_cost),
        mean_decision_latency_ms=float(np.mean(latencies)),
    )


def _aggregate_q_rows(rows: list[ControlEpisode]) -> list[ControlEpisode]:
    by_seed: dict[int, list[ControlEpisode]] = defaultdict(list)
    for row in rows:
        by_seed[row.environment_seed].append(row)
    output = []
    for environment_seed, selected in sorted(by_seed.items()):
        output.append(
            ControlEpisode(
                method="q_learning",
                environment_seed=environment_seed,
                model_seed=None,
                total_cost=float(np.mean([row.total_cost for row in selected])),
                mean_decision_latency_ms=float(
                    np.mean([row.mean_decision_latency_ms for row in selected])
                ),
            )
        )
    return output


def run_control_benchmark(
    *,
    environment_seeds: tuple[int, ...] = tuple(range(1000, 1020)),
    model_seeds: tuple[int, ...] = (0, 1, 2),
    episodes: int = 4000,
) -> dict[str, object]:
    data = default_network()
    env = DynamicSupplyControl(data=data)
    _, exact_policy = solve_dynamic_program(env)

    nominal = optimize_nominal_network(demand_scenarios(seed=7, n=12), data, step=20.0)
    robust = optimize_classical_robust_network(disruption_scenarios(), data, step=20.0)
    nominal_target = np.asarray(nominal["capacity"], dtype=float)
    robust_target = np.asarray(robust["capacity"], dtype=float)

    learned = {
        seed: train_q_learning(env, seed=seed, episodes=episodes)
        for seed in model_seeds
    }

    rows: list[ControlEpisode] = []
    q_rows: list[ControlEpisode] = []
    for environment_seed in environment_seeds:
        rows.append(
            _rollout(
                env,
                method="exact_dynamic_dp",
                environment_seed=environment_seed,
                policy_fn=lambda t, c, s: int(exact_policy[t, c, s]),
            )
        )
        rows.append(
            _rollout(
                env,
                method="myopic_or",
                environment_seed=environment_seed,
                policy_fn=lambda _t, c, s: min(
                    env.feasible_actions(c),
                    key=lambda a: (env.expected_stage_cost(s, c, int(a)), int(a)),
                ),
            )
        )
        rows.append(
            _rollout(
                env,
                method="static_nominal",
                environment_seed=environment_seed,
                policy_fn=lambda _t, c, _s: _step_toward_target(env, c, nominal_target),
            )
        )
        rows.append(
            _rollout(
                env,
                method="static_robust",
                environment_seed=environment_seed,
                policy_fn=lambda _t, c, _s: _step_toward_target(env, c, robust_target),
            )
        )
        for model_seed, result in learned.items():
            q_rows.append(
                _rollout(
                    env,
                    method="q_learning",
                    environment_seed=environment_seed,
                    model_seed=model_seed,
                    policy_fn=lambda t, c, s, r=result: q_learning_action(env, r, t, c, s),
                )
            )
    rows.extend(_aggregate_q_rows(q_rows))

    summaries = []
    reference = {
        row.environment_seed: row.total_cost
        for row in rows
        if row.method == "exact_dynamic_dp"
    }
    for method in sorted({row.method for row in rows}):
        selected = [row for row in rows if row.method == method]
        costs = np.asarray([row.total_cost for row in selected], dtype=float)
        ref = np.asarray([reference[row.environment_seed] for row in selected], dtype=float)
        stats = paired_bootstrap(costs, ref, n_bootstrap=2000, seed=77)
        summaries.append(
            {
                "method": method,
                "mean_total_cost": float(np.mean(costs)),
                "mean_gap_to_dp": stats.mean_difference,
                "gap_ci_low": stats.ci_low,
                "gap_ci_high": stats.ci_high,
                "win_rate_vs_dp": stats.win_rate,
                "mean_decision_latency_ms": float(
                    np.mean([row.mean_decision_latency_ms for row in selected])
                ),
            }
        )
    return {
        "rows": rows,
        "summaries": summaries,
        "nominal_target": nominal_target,
        "robust_target": robust_target,
        "model_seeds": model_seeds,
        "environment_seeds": environment_seeds,
    }


def main() -> None:
    result = run_control_benchmark(
        environment_seeds=tuple(range(1000, 1008)),
        model_seeds=(0, 1, 2),
        episodes=1800,
    )
    print(f"static_nominal_target={result['nominal_target'].tolist()}")
    print(f"static_robust_target={result['robust_target'].tolist()}")
    for row in result["summaries"]:
        print(
            f"{row['method']},cost={row['mean_total_cost']:.3f},"
            f"gap_to_dp={row['mean_gap_to_dp']:.3f},"
            f"ci=[{row['gap_ci_low']:.3f},{row['gap_ci_high']:.3f}],"
            f"latency_ms={row['mean_decision_latency_ms']:.4f}"
        )


if __name__ == "__main__":
    main()
