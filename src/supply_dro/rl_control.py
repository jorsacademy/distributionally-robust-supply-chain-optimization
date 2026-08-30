from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import NetworkData, default_network, first_stage_cost, recourse_cost


@dataclass(frozen=True)
class DynamicControlConfig:
    horizon: int = 6
    capacity_step: float = 20.0
    max_adjustment: float = 20.0
    adjustment_cost: float = 0.35
    gamma: float = 1.0


@dataclass(frozen=True)
class QLearningResult:
    q_values: np.ndarray
    training_seed: int
    episodes: int


class DynamicSupplyControl:
    """Small finite MDP for sequential capacity control.

    The observed signal is either ``0`` (normal) or ``1`` (stress). The action
    selects the next supplier-capacity vector subject to a one-grid-step change
    limit. Demand and availability are then sampled from a signal-dependent
    finite distribution, recourse is solved by the repository's LP, and the
    signal transitions according to a two-state Markov chain.
    """

    def __init__(
        self,
        data: NetworkData | None = None,
        config: DynamicControlConfig | None = None,
    ) -> None:
        self.data = default_network() if data is None else data
        self.config = DynamicControlConfig() if config is None else config
        if self.config.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.config.capacity_step <= 0.0:
            raise ValueError("capacity_step must be positive")
        if self.config.max_adjustment < 0.0:
            raise ValueError("max_adjustment must be nonnegative")

        self.levels = np.arange(
            0.0,
            float(np.max(self.data.supplier_capacity)) + 0.5 * self.config.capacity_step,
            self.config.capacity_step,
        )
        self.capacity_actions = np.asarray(
            [(a, b) for a in self.levels for b in self.levels], dtype=float
        )
        self.signal_transition = np.asarray([[0.82, 0.18], [0.35, 0.65]], dtype=float)
        self._outcomes = {
            0: self._combine_outcomes(
                demands=[(np.array([42.0, 38.0]), 0.25), (np.array([48.0, 42.0]), 0.50), (np.array([54.0, 46.0]), 0.25)],
                availability=[(np.array([1.0, 1.0]), 0.70), (np.array([0.85, 1.0]), 0.15), (np.array([1.0, 0.85]), 0.15)],
            ),
            1: self._combine_outcomes(
                demands=[(np.array([50.0, 45.0]), 0.20), (np.array([58.0, 50.0]), 0.50), (np.array([66.0, 56.0]), 0.30)],
                availability=[(np.array([0.65, 1.0]), 0.30), (np.array([1.0, 0.65]), 0.30), (np.array([0.70, 0.75]), 0.40)],
            ),
        }
        self._operating_cost = self._build_operating_cost_table()

    @staticmethod
    def _combine_outcomes(
        demands: list[tuple[np.ndarray, float]],
        availability: list[tuple[np.ndarray, float]],
    ) -> tuple[tuple[np.ndarray, np.ndarray, float], ...]:
        rows = []
        for demand, p_demand in demands:
            for avail, p_avail in availability:
                rows.append((demand.copy(), avail.copy(), float(p_demand * p_avail)))
        total = sum(prob for _, _, prob in rows)
        if abs(total - 1.0) > 1e-10:
            raise ValueError("outcome probabilities must sum to one")
        return tuple(rows)

    @property
    def n_capacity_states(self) -> int:
        return len(self.capacity_actions)

    def capacity_index(self, capacity: np.ndarray) -> int:
        capacity = np.asarray(capacity, dtype=float)
        matches = np.where(np.all(np.isclose(self.capacity_actions, capacity), axis=1))[0]
        if len(matches) != 1:
            raise ValueError("capacity is not on the control grid")
        return int(matches[0])

    def feasible_actions(self, capacity_index: int) -> np.ndarray:
        current = self.capacity_actions[capacity_index]
        diff = np.abs(self.capacity_actions - current)
        feasible = np.where(np.all(diff <= self.config.max_adjustment + 1e-12, axis=1))[0]
        return feasible.astype(int)

    def _build_operating_cost_table(self) -> np.ndarray:
        table = np.zeros((2, self.n_capacity_states, 9), dtype=float)
        for signal in (0, 1):
            outcomes = self._outcomes[signal]
            for action_index, capacity in enumerate(self.capacity_actions):
                for outcome_index, (demand, availability, _) in enumerate(outcomes):
                    table[signal, action_index, outcome_index] = first_stage_cost(
                        capacity, self.data
                    ) + recourse_cost(capacity, demand, self.data, availability=availability)
        return table

    def expected_stage_cost(self, signal: int, current_index: int, action_index: int) -> float:
        if action_index not in self.feasible_actions(current_index):
            return float("inf")
        probabilities = np.asarray([row[2] for row in self._outcomes[signal]], dtype=float)
        operating = float(probabilities @ self._operating_cost[signal, action_index])
        adjustment = self.config.adjustment_cost * float(
            np.sum(np.abs(self.capacity_actions[action_index] - self.capacity_actions[current_index]))
        )
        return operating + adjustment

    def sample_step(
        self,
        *,
        signal: int,
        current_index: int,
        action_index: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        if action_index not in self.feasible_actions(current_index):
            raise ValueError("infeasible capacity adjustment")
        probabilities = np.asarray([row[2] for row in self._outcomes[signal]], dtype=float)
        outcome_index = int(rng.choice(len(probabilities), p=probabilities))
        operating = float(self._operating_cost[signal, action_index, outcome_index])
        adjustment = self.config.adjustment_cost * float(
            np.sum(np.abs(self.capacity_actions[action_index] - self.capacity_actions[current_index]))
        )
        next_signal = int(rng.choice(2, p=self.signal_transition[signal]))
        return operating + adjustment, next_signal


def solve_dynamic_program(env: DynamicSupplyControl) -> tuple[np.ndarray, np.ndarray]:
    """Exact finite-horizon DP oracle for the discretized control MDP."""
    h = env.config.horizon
    values = np.zeros((h + 1, env.n_capacity_states, 2), dtype=float)
    policy = np.zeros((h, env.n_capacity_states, 2), dtype=int)
    for t in range(h - 1, -1, -1):
        for current in range(env.n_capacity_states):
            feasible = env.feasible_actions(current)
            for signal in (0, 1):
                candidates = []
                for action in feasible:
                    stage = env.expected_stage_cost(signal, current, int(action))
                    future = env.config.gamma * float(
                        env.signal_transition[signal] @ values[t + 1, int(action)]
                    )
                    candidates.append((stage + future, int(action)))
                best_cost, best_action = min(candidates, key=lambda row: (row[0], row[1]))
                values[t, current, signal] = best_cost
                policy[t, current, signal] = best_action
    return values, policy


def train_q_learning(
    env: DynamicSupplyControl,
    *,
    seed: int = 0,
    episodes: int = 4000,
    alpha: float = 0.12,
    epsilon_start: float = 0.35,
    epsilon_end: float = 0.03,
) -> QLearningResult:
    """Tabular Q-learning baseline using only sampled transitions and costs."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    q = np.zeros((env.config.horizon, env.n_capacity_states, 2, env.n_capacity_states))
    midpoint = env.capacity_index(np.array([40.0, 40.0]))

    for episode in range(episodes):
        frac = episode / max(episodes - 1, 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        current = midpoint
        signal = 0
        for t in range(env.config.horizon):
            feasible = env.feasible_actions(current)
            if rng.random() < epsilon:
                action = int(rng.choice(feasible))
            else:
                action = int(feasible[np.argmin(q[t, current, signal, feasible])])
            cost, next_signal = env.sample_step(
                signal=signal,
                current_index=current,
                action_index=action,
                rng=rng,
            )
            if t + 1 == env.config.horizon:
                target = cost
            else:
                next_feasible = env.feasible_actions(action)
                target = cost + env.config.gamma * float(
                    np.min(q[t + 1, action, next_signal, next_feasible])
                )
            q[t, current, signal, action] += alpha * (
                target - q[t, current, signal, action]
            )
            current = action
            signal = next_signal
    return QLearningResult(q_values=q, training_seed=seed, episodes=episodes)


def q_learning_action(
    env: DynamicSupplyControl,
    result: QLearningResult,
    t: int,
    current_index: int,
    signal: int,
) -> int:
    feasible = env.feasible_actions(current_index)
    return int(feasible[np.argmin(result.q_values[t, current_index, signal, feasible])])
