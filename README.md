# Distributionally Robust Supply Chain Optimization

Research-oriented Industrial Engineering / Operations Research benchmark for supply-network decisions under **distribution shift, supplier disruption and sequential control**.

## Research questions

1. How should a two-stage supply-network planner trade nominal efficiency against resilience when uncertainty comes from demand distribution shift and supplier-capacity loss?
2. When capacity can be adjusted repeatedly over time, can a reinforcement-learning controller learn useful adaptive decisions relative to static stochastic/robust policies, a myopic OR controller and an exact finite-MDP dynamic-programming oracle?

## Current status

**Feature-complete research benchmark, including the sequential RL/control extension.**

The repository implements:

- two-stage supplier-capacity reservation and LP recourse;
- correlated vector demand scenarios;
- supplier availability multipliers for capacity disruptions;
- nominal empirical stochastic optimization;
- finite-support demand-only 1-Wasserstein DRO;
- classical min-max robust optimization over named demand/disruption stresses;
- joint demand-and-availability Wasserstein DRO;
- validation-only ambiguity-radius calibration;
- frozen nominal, demand-shift and joint-shift final evaluation blocks;
- a separate finite-horizon sequential capacity-control MDP;
- an exact dynamic-programming oracle for that discretized MDP;
- tabular Q-learning trained only from sampled transitions;
- static nominal, static robust and myopic-OR control baselines;
- multi-seed Q-learning aggregation, common-random-number evaluation and paired bootstrap intervals;
- reproducible configs, tests, final report and CI across Python 3.10–3.12.

## Two-stage formulation

For reserved supplier capacity `x`, realized demand `d` and supplier availability `a`, the recourse problem chooses shipment `y_sm` and shortage `u_m`:

```text
min  sum(c_sm y_sm) + sum(p_m u_m)
s.t. sum_m y_sm <= a_s x_s                  for each supplier s
     sum_s y_sm + u_m >= d_m                 for each market m
     y_sm, u_m >= 0
```

Reservation cost is paid before uncertainty is observed.

## Static uncertainty paradigms

### Nominal stochastic optimization

```text
reservation_cost(x) + empirical_mean[recourse(x, d)]
```

### Demand-only Wasserstein DRO

```text
reservation_cost(x) + worst_case_Q E_Q[recourse(x, d)]
```

`Q` lies in a finite-support 1-Wasserstein ball around empirical vector-demand scenarios. Supplier availability is held at full availability.

### Joint demand/disruption Wasserstein DRO

The ambiguity state contains both demand and supplier availability. The transport metric combines normalized L1 demand distance with a separately weighted availability distance, allowing the adversary to shift probability mass toward states that are both demand-intensive and capacity-impaired.

### Classical finite-set robust optimization

```text
min_x max_scenario total_cost(x, demand_s, availability_s)
```

The stress set contains demand surges, individual supplier disruptions and a joint stress case.

These paradigms remain separate because their uncertainty assumptions are materially different.

## Sequential RL/control extension

The RL extension does **not** modify the frozen two-stage benchmark. It defines a separate finite-horizon control problem that reuses the same network recourse LP.

At each period the controller observes:

- the current supplier-capacity vector on a discrete grid;
- a public supply-demand regime signal: `normal` or `stress`.

The action chooses the next capacity vector subject to a one-grid-step adjustment limit per supplier. Then demand and supplier availability are sampled from a signal-dependent finite distribution. Period cost is:

```text
capacity reservation cost
+ capacity adjustment cost
+ LP recourse cost
```

The regime evolves according to a two-state Markov chain.

### RL and control baselines

The sequential benchmark compares:

- `exact_dynamic_dp`: exact backward DP on the discretized MDP;
- `myopic_or`: minimizes expected one-period cost and ignores future regime transitions;
- `static_nominal`: moves toward the nominal stochastic capacity target;
- `static_robust`: moves toward the classical-robust capacity target;
- `q_learning`: tabular Q-learning using sampled transitions only.

Q-learning never receives the exact DP value function or future random outcomes. The DP solution is an evaluation oracle, not an expert-training signal.

Three independent Q-learning training seeds are evaluated and averaged **within each environment seed** before statistical comparison, so model seeds are not incorrectly counted as independent test instances.

## Radius calibration and leakage control

Candidate Wasserstein radii are selected using **validation data only**. The selection criterion is validation p90 total cost. `configs/final_evaluation.json` freezes the train, validation and final blocks plus the candidate radius grid.

The sequential RL benchmark has its own frozen contract in `configs/rl_control.json`; final environment seeds are not used to train Q-learning.

## Frozen evaluations

`python -m supply_dro.final_campaign` evaluates the static uncertainty policies on identical paired realizations in:

- `nominal_final`;
- `demand_shift`;
- `joint_shift`.

`python -m supply_dro.rl_experiment` evaluates the sequential controllers with common random numbers and reports:

- mean total cost;
- mean cost gap to exact dynamic DP;
- paired bootstrap 95% interval;
- paired win rate versus DP;
- mean decision latency;
- static nominal and robust capacity targets.

The objective is not to force Q-learning to win. If myopic OR or a static robust policy is competitive, that result is retained.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m supply_dro.experiment
python -m supply_dro.final_campaign
python -m supply_dro.rl_experiment
```

## Repository map

```text
src/supply_dro/
  model.py
  wasserstein.py
  optimize.py
  network.py
  network_wasserstein.py
  network_optimize.py
  joint_wasserstein.py
  final_methods.py
  statistics.py
  experiment.py
  final_campaign.py
  rl_control.py
  rl_experiment.py
tests/
  test_dro.py
  test_network_dro.py
  test_disruptions.py
  test_final_campaign.py
  test_rl_control.py
configs/
  experiment.json
  final_evaluation.json
  rl_control.json
docs/
  final_report.md
.github/workflows/
  ci.yml
```

## Validation contract

A result is accepted only if:

1. demand-only `epsilon = 0` reproduces empirical expected recourse;
2. joint `epsilon = 0` reproduces empirical recourse under aligned demand/availability states;
3. supplier availability remains in `[0, 1]`;
4. all unmet demand is represented explicitly as shortage;
5. first-stage reservation cost is included in every static final total-cost metric;
6. radii are selected only from validation data;
7. every static policy is evaluated on identical final realizations;
8. nominal, demand-DRO, joint-DRO and classical-robust results remain separate;
9. sequential capacity adjustments always satisfy the control grid and adjustment limit;
10. exact DP is used only as a control oracle/evaluation reference, not as Q-learning supervision;
11. Q-learning model seeds are aggregated within environment instances before paired inference;
12. complex methods must earn their place against simpler OR/control baselines.

## Scope boundary

This repository is complete as a small uncertainty-and-control benchmark. The static two-stage study and sequential RL study are intentionally separate layers sharing the same recourse model. Multi-echelon networks, facility-location binaries, continuous-action RL, endogenous disruptions, lead-time state and large-network decomposition should be separate research projects rather than further expansion of this repository.

## License

MIT
