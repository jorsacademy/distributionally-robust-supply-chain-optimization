# Distributionally Robust Supply Chain Optimization

Research-oriented Industrial Engineering / Operations Research benchmark for two-stage supply-network decisions under **distribution shift and supplier disruption**.

## Research question

How should a supply-network planner trade nominal efficiency against resilience when uncertainty comes from both demand distribution shift and supplier-capacity loss, and when does joint distributional robustness add value beyond demand-only DRO or a classical stress set?

## Current status

**Feature-complete research benchmark.**

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
- paired bootstrap comparisons against the nominal policy;
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

## Four uncertainty paradigms

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

The stress set contains demand surges, individual supplier disruptions and a joint stress case. This model protects against named operational scenarios rather than a probabilistic ambiguity neighborhood.

These paradigms are reported separately because their uncertainty assumptions are materially different.

## Radius calibration and test leakage control

Candidate Wasserstein radii are selected using **validation data only**. The selection criterion is validation p90 total cost. `configs/final_evaluation.json` freezes the train, validation and final blocks plus the candidate radius grid.

The final campaign never uses final scenarios for radius selection or model redesign.

## Frozen final evaluation

`python -m supply_dro.final_campaign` evaluates every policy on identical paired realizations in three blocks:

- `nominal_final`: nominal demand, full supplier availability;
- `demand_shift`: higher-demand distribution, full availability;
- `joint_shift`: shifted demand plus supplier-availability losses.

The campaign reports:

- selected capacity vector;
- calibrated demand-only and joint-DRO radii;
- mean total cost;
- p90 total cost;
- worst total cost;
- mean recourse cost;
- optimization time;
- paired mean cost difference versus nominal;
- 95% paired bootstrap interval;
- paired win rate.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m supply_dro.experiment
python -m supply_dro.final_campaign
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
tests/
  test_dro.py
  test_network_dro.py
  test_disruptions.py
  test_final_campaign.py
configs/
  experiment.json
  final_evaluation.json
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
5. first-stage reservation cost is included in every final total-cost metric;
6. radii are selected only from validation data;
7. every policy is evaluated on identical final realizations;
8. nominal, demand-DRO, joint-DRO and classical-robust results remain separate;
9. robustness is judged from cost/tail trade-offs rather than one cherry-picked metric.

See `docs/final_report.md` for the full methodological contract.

## Scope boundary

This repository is complete as a small two-stage uncertainty benchmark. Multi-echelon networks, facility-location binaries, lead-time dynamics, endogenous disruptions and large-network decomposition should be separate research extensions rather than silent changes to this frozen benchmark.

## License

MIT
