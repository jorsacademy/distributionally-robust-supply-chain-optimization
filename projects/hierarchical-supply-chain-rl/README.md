# Hierarchical Supply Chain Reinforcement Learning

A reproducible industrial-engineering benchmark for **two-timescale supply-chain control**. Tactical decisions are made weekly while operational replenishment decisions are made daily.

The repository is designed to answer a practical question: when supply-chain decisions naturally live at different planning horizons, does a hierarchical policy outperform a flat or fixed policy while maintaining service and controlling cost?

## Problem formulation

The environment models a stochastic supply network with inventory, backlog, in-transit material, time-varying demand and capacity-limited replenishment.

### State

The normalized observation contains:

- inventory,
- backlog,
- in-transit quantity,
- current demand,
- active weekly allocation target,
- week phase,
- horizon phase,
- cumulative service level.

### Hierarchical decisions

**Manager / tactical level — weekly**

Chooses a normalized allocation/capacity target.

**Worker / operational level — daily**

Chooses the daily replenishment release subject to the current tactical target.

This separation is important: the manager should not micromanage daily flow, while the worker should not redefine strategic capacity every day.

### Reward

The objective minimizes

`holding cost + backlog cost + transport cost + lost-sales cost + allocation-change cost`.

Reward is the negative of this total operating cost.

## Algorithms and baselines

The repository includes:

- fixed-capacity benchmark,
- hierarchical base-stock heuristic,
- PPO manager trained on a weekly manager environment,
- PPO worker trained on a daily replenishment environment,
- composed hierarchical PPO evaluation.

The training procedure is deliberately described as **sequential hierarchical PPO**, not as a claim of a fully end-to-end options framework. The manager is trained with a fixed worker heuristic and the worker with a deterministic tactical manager; the learned policies are then composed for evaluation.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/hierarchical_supply_chain/
│   ├── environment.py
│   ├── baselines.py
│   ├── wrappers.py
│   ├── train_hrl.py
│   └── evaluate.py
├── tests/
│   ├── test_environment.py
│   └── test_wrappers_and_baselines.py
└── .github/workflows/ci.yml
```

## Installation

Core environment and baselines:

```bash
pip install -e '.[dev]'
```

RL training:

```bash
pip install -e '.[rl,dev]'
```

## Evaluate classical policies

```bash
python -m hierarchical_supply_chain.evaluate --episodes 30
```

The evaluator reports:

- mean return,
- mean total cost,
- service level,
- final inventory,
- final backlog.

## Train hierarchical PPO

```bash
python -m hierarchical_supply_chain.train_hrl \
  --manager-timesteps 40000 \
  --worker-timesteps 80000
```

The script saves:

- `artifacts/manager_ppo.zip`
- `artifacts/worker_ppo.zip`

## Evaluate learned hierarchy

```bash
python -m hierarchical_supply_chain.evaluate \
  --episodes 50 \
  --manager-model artifacts/manager_ppo.zip \
  --worker-model artifacts/worker_ppo.zip
```

## Experimental design

For a defensible comparison:

1. train on one demand distribution,
2. evaluate all policies on identical unseen seeds,
3. report both service and cost,
4. test demand surges and capacity shortages,
5. compare hierarchical policies with a flat PPO benchmark,
6. run ablations on manager decision frequency.

## Industrial engineering relevance

This project combines:

- inventory control,
- tactical allocation,
- operational replenishment,
- service-level management,
- stochastic simulation,
- hierarchical decision making,
- reinforcement learning.

The central IE question is not simply whether RL earns more reward, but whether separating tactical and operational control produces more stable and interpretable decisions under uncertainty.

## Research extensions

Useful extensions include:

- multi-echelon networks,
- multiple suppliers and facilities,
- lead-time uncertainty,
- transport mode selection,
- supplier disruptions,
- multi-agent decentralized replenishment,
- options / option-critic HRL,
- goal-conditioned worker policies,
- joint end-to-end manager-worker training,
- MILP rolling-horizon baseline,
- robust optimization under forecast error,
- carbon-emission objectives,
- CVaR and service-level constraints.

## CI

GitHub Actions runs unit tests and a short classical-policy evaluation on Python 3.10, 3.11 and 3.12. Long PPO training is intentionally excluded from CI.
