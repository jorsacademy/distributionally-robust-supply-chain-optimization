import numpy as np

from hierarchical_supply_chain.environment import HierarchicalSupplyChainEnv


def test_reset_is_seeded_and_observation_valid():
    env = HierarchicalSupplyChainEnv()
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs1)
    np.testing.assert_allclose(obs1, obs2)


def test_tactical_target_changes_only_at_week_boundary():
    env = HierarchicalSupplyChainEnv(week_length=7)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([0.8, 0.4], dtype=np.float32))
    assert np.isclose(info["weekly_target"], 0.8)
    _, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
    assert np.isclose(info["weekly_target"], 0.8)


def test_operational_release_respects_tactical_capacity():
    env = HierarchicalSupplyChainEnv(capacity=100.0)
    env.reset(seed=2)
    _, _, _, _, info = env.step(np.array([0.3, 1.0], dtype=np.float32))
    assert info["release"] <= 30.0 + 1e-5


def test_episode_truncates_at_horizon():
    env = HierarchicalSupplyChainEnv(horizon_days=5)
    env.reset(seed=3)
    truncated = False
    for _ in range(5):
        _, _, _, truncated, _ = env.step(np.array([0.6, 0.6], dtype=np.float32))
    assert truncated
