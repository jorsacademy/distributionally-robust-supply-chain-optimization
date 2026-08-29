from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .final_methods import calibrate_demand_radius, calibrate_joint_radius
from .network import (
    default_network,
    demand_scenarios,
    disruption_scenarios,
    first_stage_cost,
    recourse_cost,
)
from .network_optimize import optimize_classical_robust_network, optimize_nominal_network
from .statistics import paired_bootstrap


@dataclass(frozen=True)
class FinalRow:
    block: str
    scenario_id: int
    policy: str
    total_cost: float
    first_stage_cost: float
    recourse_cost: float


def availability_scenarios(
    seed: int,
    n: int,
    n_suppliers: int = 2,
    disruption_probability: float = 0.25,
) -> np.ndarray:
    """Generate deterministic supplier-availability scenarios from a seed."""
    rng = np.random.default_rng(seed)
    availability = np.ones((n, n_suppliers), dtype=float)
    for row in range(n):
        for supplier in range(n_suppliers):
            if rng.random() < disruption_probability:
                availability[row, supplier] = float(rng.uniform(0.30, 0.75))
    return availability


def _evaluate_capacity(
    capacity: np.ndarray,
    demand: np.ndarray,
    availability: np.ndarray,
    policy: str,
    block: str,
    data,
) -> list[FinalRow]:
    first = first_stage_cost(capacity, data)
    rows = []
    for scenario_id in range(len(demand)):
        recourse = recourse_cost(
            capacity,
            demand[scenario_id],
            data,
            availability=availability[scenario_id],
        )
        rows.append(
            FinalRow(
                block=block,
                scenario_id=scenario_id,
                policy=policy,
                total_cost=first + recourse,
                first_stage_cost=first,
                recourse_cost=recourse,
            )
        )
    return rows


def run_final_campaign(
    *,
    train_seed: int = 7,
    validation_seed: int = 37,
    final_seed: int = 107,
    train_n: int = 12,
    validation_n: int = 10,
    final_n: int = 24,
    radii: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0),
    grid_step: float = 20.0,
) -> dict[str, object]:
    """Run frozen validation-only calibration and untouched final evaluation."""
    data = default_network()
    train_demand = demand_scenarios(seed=train_seed, n=train_n)
    train_availability = availability_scenarios(train_seed + 1, train_n)
    validation_demand = demand_scenarios(seed=validation_seed, n=validation_n, shift=5.0)
    validation_availability = availability_scenarios(validation_seed + 1, validation_n, disruption_probability=0.30)

    start = perf_counter()
    nominal = optimize_nominal_network(train_demand, data, step=grid_step)
    nominal_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    demand_calibration = calibrate_demand_radius(
        train_demand,
        validation_demand,
        data,
        radii,
        step=grid_step,
    )
    demand_dro_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    joint_calibration = calibrate_joint_radius(
        train_demand,
        train_availability,
        validation_demand,
        validation_availability,
        data,
        radii,
        step=grid_step,
    )
    joint_dro_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    classical = optimize_classical_robust_network(
        disruption_scenarios(),
        data,
        step=grid_step,
    )
    classical_ms = (perf_counter() - start) * 1000.0

    capacities = {
        "nominal": np.asarray(nominal["capacity"], dtype=float),
        "demand_dro": np.asarray(demand_calibration["decision"]["capacity"], dtype=float),
        "joint_dro": np.asarray(joint_calibration["decision"]["capacity"], dtype=float),
        "classical_robust": np.asarray(classical["capacity"], dtype=float),
    }

    blocks = {
        "nominal_final": (
            demand_scenarios(seed=final_seed, n=final_n),
            np.ones((final_n, len(data.supplier_capacity))),
        ),
        "demand_shift": (
            demand_scenarios(seed=final_seed + 100, n=final_n, shift=12.0),
            np.ones((final_n, len(data.supplier_capacity))),
        ),
        "joint_shift": (
            demand_scenarios(seed=final_seed + 200, n=final_n, shift=9.0),
            availability_scenarios(
                final_seed + 201,
                final_n,
                disruption_probability=0.40,
            ),
        ),
    }

    rows: list[FinalRow] = []
    for block, (demand, availability) in blocks.items():
        for policy, capacity in capacities.items():
            rows.extend(_evaluate_capacity(capacity, demand, availability, policy, block, data))

    return {
        "rows": rows,
        "capacities": capacities,
        "demand_epsilon": float(demand_calibration["epsilon"]),
        "joint_epsilon": float(joint_calibration["epsilon"]),
        "validation_demand_p90": float(demand_calibration["validation_p90"]),
        "validation_joint_p90": float(joint_calibration["validation_p90"]),
        "solve_time_ms": {
            "nominal": nominal_ms,
            "demand_dro": demand_dro_ms,
            "joint_dro": joint_dro_ms,
            "classical_robust": classical_ms,
        },
    }


def summarize(rows: list[FinalRow]) -> list[dict[str, float | str]]:
    output = []
    for block, policy in sorted({(row.block, row.policy) for row in rows}):
        selected = [row for row in rows if row.block == block and row.policy == policy]
        costs = np.asarray([row.total_cost for row in selected])
        output.append(
            {
                "block": block,
                "policy": policy,
                "mean_total_cost": float(np.mean(costs)),
                "p90_total_cost": float(np.quantile(costs, 0.90)),
                "worst_total_cost": float(np.max(costs)),
                "mean_recourse_cost": float(np.mean([row.recourse_cost for row in selected])),
            }
        )
    return output


def paired_against_nominal(rows: list[FinalRow]) -> list[dict[str, float | str]]:
    output = []
    for block in sorted({row.block for row in rows}):
        reference_rows = sorted(
            [row for row in rows if row.block == block and row.policy == "nominal"],
            key=lambda row: row.scenario_id,
        )
        reference = np.asarray([row.total_cost for row in reference_rows])
        for policy in ("demand_dro", "joint_dro", "classical_robust"):
            candidate_rows = sorted(
                [row for row in rows if row.block == block and row.policy == policy],
                key=lambda row: row.scenario_id,
            )
            candidate = np.asarray([row.total_cost for row in candidate_rows])
            result = paired_bootstrap(candidate, reference)
            output.append(
                {
                    "block": block,
                    "policy": policy,
                    "mean_difference": result.mean_difference,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "win_rate": result.win_rate,
                    "n": float(result.n),
                }
            )
    return output


def main() -> None:
    result = run_final_campaign()
    print(f"calibrated_demand_epsilon={result['demand_epsilon']:.3f}")
    print(f"calibrated_joint_epsilon={result['joint_epsilon']:.3f}")
    print("block,policy,mean_total,p90_total,worst_total,mean_recourse")
    for row in summarize(result["rows"]):
        print(
            f"{row['block']},{row['policy']},{row['mean_total_cost']:.3f},"
            f"{row['p90_total_cost']:.3f},{row['worst_total_cost']:.3f},"
            f"{row['mean_recourse_cost']:.3f}"
        )
    print("block,policy,mean_diff_vs_nominal,ci_low,ci_high,win_rate,n")
    for row in paired_against_nominal(result["rows"]):
        print(
            f"{row['block']},{row['policy']},{row['mean_difference']:.3f},"
            f"{row['ci_low']:.3f},{row['ci_high']:.3f},{row['win_rate']:.3f},"
            f"{int(row['n'])}"
        )


if __name__ == "__main__":
    main()
