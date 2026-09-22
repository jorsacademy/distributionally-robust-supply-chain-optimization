"""Train manager and worker PPO policies on their native timescales."""

from __future__ import annotations

import argparse
from pathlib import Path

from .wrappers import DailyReplenishmentEnv, WeeklyAllocationEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manager-timesteps", type=int, default=40000)
    parser.add_argument("--worker-timesteps", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manager = PPO("MlpPolicy", WeeklyAllocationEnv(), verbose=1, seed=args.seed)
    manager.learn(total_timesteps=args.manager_timesteps)
    manager.save(args.output_dir / "manager_ppo")

    worker = PPO("MlpPolicy", DailyReplenishmentEnv(), verbose=1, seed=args.seed + 1)
    worker.learn(total_timesteps=args.worker_timesteps)
    worker.save(args.output_dir / "worker_ppo")

    print(f"Saved manager and worker models to {args.output_dir}")


if __name__ == "__main__":
    main()
