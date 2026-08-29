# Distributionally Robust Supply Chain Optimization

Research-oriented Industrial Engineering / Operations Research project for supply-chain decisions under **distribution shift and supplier disruption** using two-stage stochastic optimization, finite-support Wasserstein DRO and a classical finite stress-set robust baseline.

## Research question

How should a supply-network planner trade nominal efficiency against resilience when uncertainty comes from both demand distribution shift and supplier-capacity loss?

## Current status

**Phase 3 implemented: two-stage network + Wasserstein DRO + supplier disruption + classical robust comparison.**

The repository contains:

- two suppliers and two demand markets;
- first-stage supplier-capacity reservation decisions;
- supplier capacity limits;
- market-specific shortage penalties;
- second-stage shipment and shortage recourse solved by LP;
- correlated vector demand scenarios;
- finite-support 1-Wasserstein ambiguity over demand scenarios;
- supplier availability multipliers for disruption states;
- a finite demand/disruption stress set;
- nominal expected-value capacity optimization;
- Wasserstein DRO capacity optimization;
- classical min-max robust capacity optimization over the stress set;
- independent OOD demand-shift evaluation;
- disruption stress evaluation;
- tests and CI across Python 3.10-3.12.

## Two-stage formulation

For reserved supplier capacity `x`, demand `d` and supplier availability `a`, the recourse problem chooses shipment `y_sm` and shortage `u_m`:

```text
min  sum(c_sm y_sm) + sum(p_m u_m)
s.t. sum_m y_sm <= a_s x_s                  for each supplier s
     sum_s y_sm + u_m >= d_m                 for each market m
     y_sm, u_m >= 0
```

The first stage pays reservation cost before uncertainty is realized.

## Three decision paradigms

### Nominal stochastic optimization

```text
reservation_cost(x) + empirical_mean[recourse(x, d)]
```

This is efficient when the training distribution is representative but does not directly protect against distribution shift or supplier outages.

### Wasserstein DRO

```text
reservation_cost(x) + worst_case_Q E_Q[recourse(x, d)]
```

`Q` lies inside a finite-support 1-Wasserstein ball around empirical vector-demand scenarios. This model perturbs the probability law of demand while retaining the same recourse structure.

### Classical finite-set robust optimization

```text
min_x max_scenario total_cost(x, demand_s, availability_s)
```

The stress set includes demand surges, individual supplier capacity losses and a joint stress case. This baseline protects explicitly against named operational disruptions rather than probabilistic distribution shift.

The repository intentionally keeps these two notions of robustness separate.

## Supplier disruption model

Supplier availability is represented by a multiplier in `[0, 1]`. A value of `0.35`, for example, means only 35% of reserved capacity is usable in that scenario. Reservation cost is still incurred because the first-stage decision is made before the disruption is known.

## Validation invariants

The implementation checks that:

1. Wasserstein radius `epsilon = 0` reproduces empirical expected recourse;
2. worst-case Wasserstein recourse is non-decreasing with radius;
3. disruption cannot improve recourse cost merely by removing usable supply capacity;
4. first-stage capacity remains within supplier limits;
5. recourse explicitly accounts for unmet demand through shortage variables;
6. nominal, DRO and classical robust policies are evaluated on common OOD/stress sets.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m supply_dro.experiment
```

The experiment reports, for each policy family:

- selected supplier capacity vector;
- training mean total cost;
- OOD mean, p90 and worst total cost;
- mean stress-set cost;
- worst stress-set cost;
- identity of the binding worst disruption scenario.

## Repository map

```text
src/supply_dro/
  model.py                 # original single-period reference model
  wasserstein.py           # scalar-demand Wasserstein transport LP
  optimize.py              # original reference optimizer
  network.py               # two-stage network, disruptions and recourse LP
  network_wasserstein.py   # vector-demand Wasserstein adversary
  network_optimize.py      # nominal, DRO and classical robust policies
  experiment.py            # OOD + disruption comparison campaign
tests/
  test_dro.py
  test_network_dro.py
  test_disruptions.py
configs/
  experiment.json
.github/workflows/
  ci.yml
```

## Evaluation contract

Do not declare a robust method superior from one metric. Report nominal cost, OOD distribution-shift performance, worst stress cost, tail metrics, capacity commitment and the uncertainty model used. A policy that is more expensive nominally may be justified only if the resilience gain is measurable.

## Next research stages

### Phase 4 — ambiguity-radius calibration
Use validation-only data to select Wasserstein radius. Freeze the radius before final-test evaluation.

### Phase 5 — joint demand/disruption ambiguity
Extend the ambiguity state beyond vector demand so supplier availability itself becomes uncertain inside a joint distributional model.

### Phase 6 — larger network
Add more suppliers, warehouses and customer zones. Replace capacity-grid enumeration with an optimization/decomposition method once dimensionality makes grid search inappropriate.

### Phase 7 — statistical final campaign
Use frozen validation/final scenario blocks, paired comparisons, bootstrap confidence intervals, CVaR, service-level and shortage-rate metrics.

## Portfolio signal

The project now distinguishes three materially different uncertainty paradigms—empirical stochastic optimization, distributionally robust optimization and scenario-set robust optimization—under one recourse model. That comparison is more informative than simply adding safety stock or labeling one policy as “robust.”

## License

MIT
