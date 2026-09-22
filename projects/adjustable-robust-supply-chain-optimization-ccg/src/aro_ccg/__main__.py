from __future__ import annotations

import argparse

import numpy as np

from .baselines import compare_methods
from .data import demo_instance
from .evaluation import audit_solution
from .experiments import format_sensitivity_table, run_sensitivity


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage adjustable robust supply-chain optimization with C&CG."
    )
    parser.add_argument("--gamma-demand", type=float, default=2.0)
    parser.add_argument("--gamma-disruption", type=float, default=1.0)
    parser.add_argument(
        "--oracle",
        choices=("auto", "milp", "enumeration"),
        default="auto",
        help="Adversarial subproblem method. auto uses MILP for integer budgets.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the default Gamma_demand x Gamma_disruption sensitivity grid.",
    )
    args = parser.parse_args()

    instance = demo_instance()
    comparison = compare_methods(
        instance,
        gamma_demand=args.gamma_demand,
        gamma_disruption=args.gamma_disruption,
        oracle_method=args.oracle,
    )
    result = comparison.adjustable_robust

    audit = audit_solution(
        instance,
        result.open_decision,
        result.reserved_capacity,
        gamma_demand=args.gamma_demand,
        gamma_disruption=args.gamma_disruption,
        oracle_method="enumeration",
    )

    print("Adjustable Robust Supply Chain Optimization with C&CG")
    print("=====================================================")
    print(f"Gamma demand: {args.gamma_demand:g}")
    print(f"Gamma disruption: {args.gamma_disruption:g}")
    print()
    print("Method comparison")
    print(
        f"Deterministic objective:       {comparison.deterministic.objective:12.6f}"
    )
    print(
        f"Static robust objective:       {comparison.static_robust.objective:12.6f}"
    )
    print(
        f"Adjustable robust objective:   {result.robust_objective:12.6f}"
    )
    print(
        f"Robustness premium vs nominal: {comparison.robustness_premium:12.6f}"
    )
    print(
        f"Value of adjustability:        {comparison.value_of_adjustability:12.6f}"
    )
    print()
    print(f"Converged: {result.converged}")
    print(f"Adversarial oracle: {result.oracle_method}")
    print(f"Final lower bound: {result.lower_bound:.6f}")
    print(f"Final upper bound: {result.upper_bound:.6f}")
    print(f"Open decision: {_format_vector(result.open_decision)}")
    print(f"Reserved capacity: {_format_vector(result.reserved_capacity)}")
    print(f"Master scenarios: {result.master_scenario_count}")
    print(f"Worst demand: {_format_vector(audit.worst_demand)}")
    print(f"Worst availability: {_format_vector(audit.worst_availability)}")
    print(f"Worst shortage: {_format_vector(audit.worst_shortage)}")
    print(f"Independent enumeration audit: {audit.robust_total_cost:.6f}")
    print()
    print("Iteration history")
    print("iter  scen      lower_bound      upper_bound       gap   shortage  oracle")
    for item in result.iterations:
        gap = item.upper_bound - item.lower_bound
        print(
            f"{item.iteration:>4d}"
            f"{item.scenario_count:>6d}"
            f"{item.lower_bound:>17.6f}"
            f"{item.upper_bound:>17.6f}"
            f"{gap:>12.6f}"
            f"{item.worst_total_shortage:>10.3f}  "
            f"{item.oracle_method}"
        )

    if args.sweep:
        print()
        print("Sensitivity sweep")
        records = run_sensitivity(
            instance,
            demand_gammas=[0.0, 1.0, 2.0, 3.0, 4.0],
            disruption_gammas=[0.0, 1.0, 2.0, 3.0],
            oracle_method=args.oracle,
        )
        print(format_sensitivity_table(records))


if __name__ == "__main__":
    main()
