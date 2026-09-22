from __future__ import annotations

import argparse
import json

from .data import demo_instance
from .experiment import gamma_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve and stress-test a Bertsimas-Sim robust supply-chain network design model."
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0, 3.0, 4.0],
        help="Budget-of-uncertainty values to solve.",
    )
    parser.add_argument("--scenarios", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shock-probability", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = gamma_sweep(
        demo_instance(),
        gammas=args.gammas,
        n_scenarios=args.scenarios,
        seed=args.seed,
        shock_probability=args.shock_probability,
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
