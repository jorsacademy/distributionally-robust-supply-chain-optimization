# Distributionally Robust Supply Chain Optimization

Research-oriented Industrial Engineering / Operations Research project for supply-chain decisions under **distribution shift** using two-stage stochastic optimization and finite-support Wasserstein distributionally robust optimization (DRO).

## Research question

How much nominal performance should a supply-network planner sacrifice to obtain first-stage capacity decisions that remain reliable when the future demand distribution differs from the empirical training sample?

## Current status

**Phase 2 implemented: two-stage capacitated supply network + Wasserstein ambiguity.**

The repository now contains both the original auditable single-period model and a richer network model with:

- two suppliers and two demand markets;
- first-stage supplier-capacity reservation decisions;
- supplier capacity limits;
- market-specific shortage penalties;
- second-stage shipment and shortage recourse solved by LP;
- correlated vector demand scenarios;
- finite-support 1-Wasserstein ambiguity over those scenarios;
- nominal expected-value capacity optimization;
- DRO capacity optimization;
- independent OOD demand-shift evaluation;
- tests for Wasserstein and capacity-model invariants;
- CI across supported Python versions.

## Two-stage formulation

For reserved supplier capacity `x`, each realized demand scenario `d` triggers a recourse LP. The second stage chooses shipments `y_sm` and shortages `u_m`:

```text
min  sum(c_sm y_sm) + sum(p_m u_m)
s.t. sum_m y_sm <= x_s                    for each supplier s
     sum_s y_sm + u_m >= d_m               for each market m
     y_sm, u_m >= 0
```

The first stage pays capacity reservation cost before demand is observed. Nominal optimization minimizes

```text
reservation_cost(x) + empirical_mean[recourse(x, d)].
```

The DRO model instead minimizes

```text
reservation_cost(x) + worst_case_Q E_Q[recourse(x, d)]
```

where `Q` lies inside a finite-support 1-Wasserstein ball around the empirical demand distribution.

## Wasserstein adversary

The empirical scenarios are vector-valued market demands. The adversary transports probability mass between observed scenarios subject to an L1 transportation budget. Destination scenario costs are the optimized second-stage recourse values for the fixed first-stage capacity decision.

At radius `epsilon = 0`, the worst-case recourse must equal the empirical expected recourse. Increasing the radius cannot decrease the worst-case expected recourse; both properties are tested.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m supply_dro.experiment
```

The smoke experiment trains capacity decisions on a small correlated demand sample and evaluates them on an independent distribution-shift scenario set with higher demand.

## Repository map

```text
src/supply_dro/
  model.py                 # original single-period reference model
  wasserstein.py           # scalar-demand Wasserstein transport LP
  optimize.py              # original nominal/DRO reference optimizer
  network.py               # two-stage capacitated supply network + recourse LP
  network_wasserstein.py   # vector-demand Wasserstein adversary
  network_optimize.py      # nominal and DRO first-stage capacity optimization
  experiment.py            # nominal vs DRO OOD comparison
tests/
  test_dro.py
  test_network_dro.py
configs/
  experiment.json
.github/workflows/
  ci.yml
```

## Evaluation contract

Do not select a policy on nominal expected cost alone. Report at minimum:

- training-distribution mean total cost;
- OOD mean total cost;
- OOD p90 cost;
- worst observed OOD cost;
- selected supplier capacity vector;
- sensitivity to Wasserstein radius;
- first-stage cost versus recourse-cost decomposition.

For larger experiments, add CVaR, service-level and shortage-rate metrics by market.

## Validation rules

A DRO implementation is accepted only if:

1. `epsilon = 0` reproduces empirical expected recourse;
2. worst-case expected recourse is non-decreasing in Wasserstein radius;
3. capacity decisions obey supplier limits;
4. all second-stage demand is either shipped or represented as explicit shortage;
5. nominal and DRO policies are evaluated on identical OOD scenarios.

## Next research stages

### Phase 3 — disruption uncertainty
Add supplier availability states, capacity-loss events and transport-lane disruption scenarios. Compare demand-only ambiguity against joint demand/disruption ambiguity.

### Phase 4 — stronger optimization baselines
Compare empirical stochastic programming, box/polyhedral robust optimization and Wasserstein DRO under identical first-stage and recourse structures.

### Phase 5 — radius calibration
Use validation-only procedures to calibrate the ambiguity radius. Final-test scenarios must remain untouched until model selection is frozen.

### Phase 6 — larger network
Extend to multiple suppliers, plants/warehouses and customer zones. Replace explicit capacity-grid search with mathematical optimization or decomposition once dimensionality makes enumeration inappropriate.

## Portfolio signal

The project demonstrates the difference between uncertainty modeling and simply adding safety stock: first-stage capacity decisions are evaluated through explicit recourse optimization, while the ambiguity set changes the probability law rather than manually inflating demand.

## License

MIT
