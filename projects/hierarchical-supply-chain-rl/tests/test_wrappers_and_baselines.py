import numpy as np

from hierarchical_supply_chain.baselines import BaseStockHierarchicalPolicy, ConstantCapacityPolicy
from hierarchical_supply_chain.wrappers import DailyReplenishmentEnv, WeeklyAllocationEnv


def test_constant_policy_returns_valid_joint_action():
    action = ConstantCapacityPolicy().act(np.zeros(8, dtype=np.float32))
    assert action.shape == (2,)
    assert np.all((0.0 <= action) & (action <= 1.0))


def test_base_stock_policy_reacts_to_backlog():
    policy = BaseStockHierarchicalPolicy()
    low = np.array([0.4, 0.0, 0, 0.5, 0.5, 0, 0, 0.9], dtype=np.float32)
    high = low.copy()
    high[1] = 0.8
    assert policy.act(high)[1] > policy.act(low)[1]


def test_weekly_wrapper_advances_one_week():
    env = WeeklyAllocationEnv(weeks=2)
    env.reset(seed=4)
    _, _, _, _, _ = env.step(np.array([0.6], dtype=np.float32))
    assert env.base.day == 7


def test_daily_wrapper_accepts_scalar_action():
    env = DailyReplenishmentEnv(horizon_days=5)
    obs, _ = env.reset(seed=5)
    next_obs, _, _, _, _ = env.step(np.array([0.5], dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(next_obs)
