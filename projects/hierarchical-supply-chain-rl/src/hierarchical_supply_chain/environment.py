"""Two-timescale hierarchical supply-chain environment."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SupplyChainEconomics:
    holding_cost: float = 0.8
    backlog_cost: float = 4.0
    transport_cost: float = 0.35
    allocation_change_cost: float = 0.15
    lost_sales_cost: float = 8.0


class HierarchicalSupplyChainEnv(gym.Env):
    """Single-echelon network abstraction with weekly goals and daily execution.

    High-level action: normalized weekly allocation target in [0, 1].
    Low-level action: normalized daily replenishment release in [0, 1].

    The environment exposes both through a two-dimensional Box action. The first
    component is only applied when a new tactical week begins; otherwise the
    current weekly target remains active.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        horizon_days: int = 84,
        week_length: int = 7,
        capacity: float = 120.0,
        max_inventory: float = 220.0,
        max_backlog: float = 180.0,
        economics: SupplyChainEconomics | None = None,
    ) -> None:
        super().__init__()
        self.horizon_days = horizon_days
        self.week_length = week_length
        self.capacity = capacity
        self.max_inventory = max_inventory
        self.max_backlog = max_backlog
        self.economics = economics or SupplyChainEconomics()

        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)

        self.day = 0
        self.inventory = 0.0
        self.backlog = 0.0
        self.weekly_target = 0.5
        self.previous_weekly_target = 0.5
        self.in_transit = 0.0
        self.demand = 0.0
        self.cumulative_demand = 0.0
        self.cumulative_fulfilled = 0.0
        self.total_cost = 0.0

    def _sample_demand(self) -> float:
        seasonal = 38.0 + 10.0 * np.sin(2.0 * np.pi * (self.day % 28) / 28.0)
        return float(max(5.0, self.np_random.normal(seasonal, 6.0)))

    def _observation(self) -> np.ndarray:
        week_phase = (self.day % self.week_length) / max(1, self.week_length - 1)
        horizon_phase = self.day / max(1, self.horizon_days)
        return np.asarray(
            [
                np.clip(self.inventory / self.max_inventory, 0, 1),
                np.clip(self.backlog / self.max_backlog, 0, 1),
                np.clip(self.in_transit / self.capacity, 0, 1),
                np.clip(self.demand / 80.0, 0, 1),
                np.clip(self.weekly_target, 0, 1),
                np.clip(week_phase, 0, 1),
                np.clip(horizon_phase, 0, 1),
                np.clip(self.cumulative_fulfilled / max(1.0, self.cumulative_demand), 0, 1),
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict:
        service_level = self.cumulative_fulfilled / max(1.0, self.cumulative_demand)
        return {
            "inventory": self.inventory,
            "backlog": self.backlog,
            "weekly_target": self.weekly_target,
            "in_transit": self.in_transit,
            "service_level": service_level,
            "total_cost": self.total_cost,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.inventory = 70.0
        self.backlog = 0.0
        self.weekly_target = 0.5
        self.previous_weekly_target = 0.5
        self.in_transit = 0.0
        self.demand = self._sample_demand()
        self.cumulative_demand = 0.0
        self.cumulative_fulfilled = 0.0
        self.total_cost = 0.0
        return self._observation(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        action = np.clip(action, 0.0, 1.0)

        tactical_action = float(action[0])
        operational_action = float(action[1])

        allocation_change_cost = 0.0
        if self.day % self.week_length == 0:
            self.previous_weekly_target = self.weekly_target
            self.weekly_target = tactical_action
            allocation_change_cost = self.economics.allocation_change_cost * abs(
                self.weekly_target - self.previous_weekly_target
            ) * self.capacity

        arrivals = self.in_transit
        self.inventory = min(self.max_inventory, self.inventory + arrivals)

        daily_tactical_cap = self.weekly_target * self.capacity
        release = min(operational_action * self.capacity, daily_tactical_cap)
        self.in_transit = release

        total_demand = self.demand + self.backlog
        fulfilled = min(self.inventory, total_demand)
        self.inventory -= fulfilled
        remaining = total_demand - fulfilled
        self.backlog = min(self.max_backlog, remaining)
        lost_sales = max(0.0, remaining - self.max_backlog)

        holding_cost = self.economics.holding_cost * self.inventory
        backlog_cost = self.economics.backlog_cost * self.backlog
        transport_cost = self.economics.transport_cost * release
        lost_sales_cost = self.economics.lost_sales_cost * lost_sales
        step_cost = holding_cost + backlog_cost + transport_cost + lost_sales_cost + allocation_change_cost
        self.total_cost += step_cost

        self.cumulative_demand += self.demand
        self.cumulative_fulfilled += min(fulfilled, self.demand + self.backlog)

        reward = -step_cost
        self.day += 1
        truncated = self.day >= self.horizon_days
        terminated = False
        self.demand = self._sample_demand() if not truncated else 0.0

        info = self._info()
        info.update(
            {
                "holding_cost": holding_cost,
                "backlog_cost": backlog_cost,
                "transport_cost": transport_cost,
                "lost_sales_cost": lost_sales_cost,
                "allocation_change_cost": allocation_change_cost,
                "fulfilled": fulfilled,
                "release": release,
            }
        )
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        return (
            f"day={self.day} inventory={self.inventory:.1f} backlog={self.backlog:.1f} "
            f"target={self.weekly_target:.2f} service={self._info()['service_level']:.3f}"
        )
