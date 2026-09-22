"""Gym wrappers for sequential hierarchical training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .baselines import BaseStockHierarchicalPolicy
from .environment import HierarchicalSupplyChainEnv


class WeeklyAllocationEnv(gym.Env):
    """Manager environment: one action is a weekly allocation target.

    Each manager step simulates one week while a fixed worker policy executes
    daily replenishment decisions. This makes the tactical policy trainable on
    the correct weekly timescale.
    """

    def __init__(self, weeks: int = 12):
        super().__init__()
        self.base = HierarchicalSupplyChainEnv(horizon_days=weeks * 7)
        self.worker = BaseStockHierarchicalPolicy()
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)

    def _manager_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray([obs[0], obs[1], obs[2], obs[3], obs[4], obs[7]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed)
        return self._manager_obs(obs), info

    def step(self, action):
        tactical = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        weekly_reward = 0.0
        terminated = truncated = False
        info = {}
        obs = self.base._observation()
        for _ in range(self.base.week_length):
            worker_action = self.worker.act(obs)
            joint_action = np.asarray([tactical, worker_action[1]], dtype=np.float32)
            obs, reward, terminated, truncated, info = self.base.step(joint_action)
            weekly_reward += reward
            if terminated or truncated:
                break
        return self._manager_obs(obs), weekly_reward, terminated, truncated, info


class DailyReplenishmentEnv(gym.Env):
    """Worker environment: daily release under a deterministic tactical manager."""

    def __init__(self, horizon_days: int = 84):
        super().__init__()
        self.base = HierarchicalSupplyChainEnv(horizon_days=horizon_days)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self.base.observation_space

    @staticmethod
    def manager_target(obs: np.ndarray) -> float:
        inventory, backlog, _, demand, *_ = obs
        return float(np.clip(0.45 + 0.65 * backlog + 0.25 * demand - 0.25 * inventory, 0.2, 1.0))

    def reset(self, *, seed=None, options=None):
        return self.base.reset(seed=seed)

    def step(self, action):
        obs = self.base._observation()
        tactical = self.manager_target(obs)
        operational = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return self.base.step(np.asarray([tactical, operational], dtype=np.float32))
