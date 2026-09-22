# Agent-Based Supply-Chain Resilience Simulation and Policy Optimization

A disruption-aware multi-agent supply-chain simulation with decentralized inventory policies, transport delays, supplier-capacity outages, bullwhip measurement, resilience KPIs, and finite-grid policy optimization.

```text
physical flow
Supplier -> Factory -> Warehouse -> Retailer -> Customer

information / order flow
Customer -> Retailer -> Warehouse -> Factory -> Supplier
```

A separate `CarrierAgent` controls link-specific transport lead times and disruption delays. Retailer, Warehouse, Factory, and Supplier are `InventoryAgent` objects that use only local operational state: on-hand inventory, backlog, outstanding replenishment, and a local exponentially smoothed forecast.

## Local order-up-to policy

Each inventory agent computes:

```text
inventory position = on hand + outstanding replenishment - backlog

target inventory position
= local forecast * (planning lead time + review period + safety days)

desired order = max(0, target - inventory position)
```

The simulator coordinates event sequencing and physical shipments; it does not expose global supply-chain state to the agents.

## Stochastic disruptions

Each replication contains stochastic customer demand, weekly seasonality, a demand surge, severe supplier-capacity loss with partial recovery, random supplier derating, link-specific transport delays, and disruption-period carrier congestion. The default horizon is 120 days, with the main disruption around days 55-72.

All candidate policies in a replication receive the same scenario primitives, implementing Common Random Numbers (CRN).

## Immediate fill-rate semantics

The service KPI is an immediate customer fill rate. Old backlog is served first; only the portion of today's new demand satisfied immediately from retailer inventory contributes to today's fill numerator. Late fulfillment is therefore not misclassified as on-time service.

## Supplier capacity semantics

The Supplier procures from a finite-capacity external source. During a source disruption, only the externally confirmed quantity enters the supplier's order pipeline. Desired but unconfirmed replenishment is recomputed at the next local review rather than being recorded as completed procurement.

## Bullwhip and recovery KPIs

For each echelon:

```text
bullwhip_i = variance(stage i orders) / variance(customer demand)
```

The reported bullwhip ratio is the largest echelon amplification. Recovery starts after the disruption window and is declared when customer backlog remains below 1% of average customer demand for five consecutive days.

## Policy grid

The finite decision grid is:

```text
review period             in {1, 2} days
retailer/warehouse safety in {1, 2, 3} demand-days
factory/supplier safety   in {1, 2, 3} demand-days
forecast smoothing alpha  in {0.20, 0.40, 0.65}
```

Total: `2 * 3 * 3 * 3 = 54` policies.

Every policy is evaluated on the same selection scenarios. The selected policy is therefore the exact minimizer of the declared sample score over this finite 54-policy grid. It is not a global optimum over all possible decentralized policies.

The selection score combines operating cost, a strong penalty below 98.5% immediate fill rate, a logarithmic bullwhip penalty, and a recovery-time penalty.

## Independent validation

Selection and validation scenarios use disjoint seeds. The baseline policy uses one-day review, one demand-day of safety stock at every echelon, and forecast alpha `0.65`. Selected and baseline policies are replayed on the same validation scenarios, enabling paired CRN comparison and a paired Student-t confidence interval.

Development run with seed 42, 16 selection replications, and 36 independent validation replications:

```text
selected policy
  review period                 1 day
  retailer/warehouse safety     3 demand-days
  factory/supplier safety       1 demand-day
  forecast alpha                0.20

selected validation
  mean cost                     171.438
  immediate fill rate            99.477%
  mean customer backlog           0.238
  mean total inventory          522.071
  mean bullwhip ratio            46.528
  mean recovery time              0.056 days

baseline validation
  mean cost                     279.764
  immediate fill rate            92.486%
  mean customer backlog           4.459
  mean total inventory          525.274
  mean bullwhip ratio           313.304
  mean recovery time              1.278 days

paired cost(baseline) - cost(selected)
  mean                          108.326
  95% CI                     [96.905, 119.747]
```

These are consequences of the declared synthetic disruption model and economic coefficients, not real supply-chain savings or service-level claims.

## Validation

The regression suite checks local agent arithmetic, Carrier lead-time hand oracles, deterministic constant-demand service, scenario reproducibility, supplier confirmed-order capacity, 54-policy grid completeness, CRN reproducibility, paired confidence-interval consistency, and end-to-end short policy optimization.

Run:

```bash
python agent_based_supply_chain_simulation.py
python agent_based_supply_chain_simulation.py --self-test
python -m unittest discover -s tests -v
```

CI smoke experiment:

```bash
python agent_based_supply_chain_simulation.py \
  --selection-replications 8 \
  --validation-replications 12 \
  --seed 42
```

## Validated GitHub Actions run

The complete workflow was executed on GitHub Actions with CPython 3.12.14. The ABM self-test, all nine regression tests, and the 8-selection / 12-validation stochastic policy-optimization smoke experiment completed successfully.

GitHub-runner smoke result:

```text
selected policy
  review period                 1 day
  retailer/warehouse safety     3 demand-days
  factory/supplier safety       1 demand-day
  forecast alpha                0.20

selected validation
  mean cost                     176.614
  immediate fill rate            99.574%
  mean customer backlog           0.195
  mean total inventory          523.513
  mean bullwhip ratio            46.852
  mean recovery time              0.000 days

baseline validation
  mean cost                     278.857
  immediate fill rate            92.879%
  mean customer backlog           4.288
  mean total inventory          529.526
  mean bullwhip ratio           294.338
  mean recovery time              1.500 days

paired cost(baseline) - cost(selected)
  mean                          102.243
  95% CI                     [73.597, 130.888]
```

These CI values are measurements under the declared synthetic model and the stated random seeds; they are not real-world savings, service, or resilience claims.

## Modeling scope

This is an educational ABM, not a calibrated supply-chain digital twin. It assumes one product family, daily reviews, local order-up-to policies, fixed nominal transport lead times plus stochastic delays, one external Supplier source, and no explicit BOM, substitution, endogenous pricing, MOQ rules, or multi-customer allocation. A production application would require calibration from ERP/WMS/TMS histories, supplier reliability data, lane lead-time distributions, production/yield constraints, and structural-break validation.
