"""Evaluate classical and learned hierarchical supply-chain policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import BaseStockHierarchicalPolicy, ConstantCapacityPolicy
from .environment import HierarchicalSupplyChainEnv


def evaluate_policy(name: str, policy, episodes: int = 30, seed: int = 100) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        env = HierarchicalSupplyChainEnv()
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        final_info = {}
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rows.append(
            {
                "policy": name,
                "episode": episode,
                "return": total_reward,
                "total_cost": final_info["total_cost"],
                "service_level": final_info["service_level"],
                "final_inventory": final_info["inventory"],
                "final_backlog": final_info["backlog"],
            }
        )
    return pd.DataFrame(rows)


class LearnedHierarchicalPolicy:
    """Compose a weekly manager PPO and daily worker PPO for evaluation."""

    def __init__(self, manager, worker):
        self.manager = manager
        self.worker = worker
        self.current_target = 0.5
        self.last_week_phase = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        week_phase = float(obs[5])
        if week_phase == 0.0 or self.last_week_phase is None:
            manager_obs = np.asarray([obs[0], obs[1], obs[2], obs[3], obs[4], obs[7]], dtype=np.float32)
            action, _ = self.manager.predict(manager_obs, deterministic=True)
            self.current_target = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        worker_action, _ = self.worker.predict(obs, deterministic=True)
        operational = float(np.clip(np.asarray(worker_action).reshape(-1)[0], 0.0, 1.0))
        self.last_week_phase = week_phase
        return np.asarray([self.current_target, operational], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--manager-model", type=Path)
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/comparison.csv"))
    args = parser.parse_args()

    frames = [
        evaluate_policy("constant_capacity", ConstantCapacityPolicy(), args.episodes, args.seed),
        evaluate_policy("hierarchical_base_stock", BaseStockHierarchicalPolicy(), args.episodes, args.seed),
    ]

    if args.manager_model and args.worker_model:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc
        learned = LearnedHierarchicalPolicy(PPO.load(args.manager_model), PPO.load(args.worker_model))
        frames.append(evaluate_policy("hierarchical_ppo", learned, args.episodes, args.seed))

    results = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(
        results.groupby("policy").agg(
            mean_return=("return", "mean"),
            mean_total_cost=("total_cost", "mean"),
            mean_service_level=("service_level", "mean"),
            mean_final_inventory=("final_inventory", "mean"),
            mean_final_backlog=("final_backlog", "mean"),
        ).round(3)
    )


if __name__ == "__main__":
    main()
