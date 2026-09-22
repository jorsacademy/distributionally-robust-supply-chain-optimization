# Adjustable Robust Supply Chain Optimization with C&CG

A verification-first Operations Research benchmark for **two-stage adjustable robust supply-chain design** solved by **column-and-constraint generation (C&CG)**.

The repository isolates the value of adaptivity. Supplier activation and capacity reservation are chosen before uncertainty is observed; shipments and shortage recourse are chosen after demand and supplier availability are revealed.

## Research question

How much value does adjustable recourse provide relative to a non-adaptive robust plan when demand surges and supplier-capacity losses are only known to lie in budgeted uncertainty sets?

```text
first stage
  open suppliers + reserve capacity
             |
             v
uncertainty is revealed
  demand surge + capacity loss
             |
             v
second stage
  adaptive shipments + shortage recourse
```

## Two-stage robust model

Suppliers are indexed by `s` and markets by `m`.

First-stage decisions:

- `y_s in {0,1}`: activate supplier `s`;
- `x_s >= 0`: reserve supplier capacity.

Second-stage decisions for realized uncertainty `xi`:

- `q_sm(xi) >= 0`: shipment from supplier `s` to market `m`;
- `u_m(xi) >= 0`: unmet demand.

The robust objective is

```text
min  fixed_cost(y) + reservation_cost(x)
     + max_{xi in U} Q(x, xi)
```

where `Q(x, xi)` is the optimal recourse cost

```text
min  sum_sm shipping_cost_sm q_sm
   + sum_m shortage_penalty_m u_m

s.t.
     sum_m q_sm <= availability_s(xi) * x_s       for each supplier s
     sum_s q_sm + u_m >= demand_m(xi)             for each market m
     q_sm, u_m >= 0
```

and

```text
0 <= x_s <= capacity_max_s * y_s.
```

## Budgeted uncertainty

Demand is

```text
demand_m = nominal_demand_m + demand_deviation_m * z_m

0 <= z_m <= 1
sum_m z_m <= Gamma_demand.
```

Supplier availability is

```text
availability_s = 1 - max_availability_loss_s * w_s

0 <= w_s <= 1
sum_s w_s <= Gamma_disruption.
```

The uncertainty budgets are protection parameters, not probabilities.

## Dualized adversarial MILP

For fixed first-stage capacity, the recourse LP is dualized. With nonnegative dual variables `alpha_s` for supplier-capacity rows and `beta_m` for demand rows, the dual is

```text
max  sum_m demand_m * beta_m
   - sum_s availability_s * x_s * alpha_s

s.t.
     beta_m - alpha_s <= shipping_cost_sm
     0 <= beta_m <= shortage_penalty_m
     alpha_s >= 0.
```

For **integer** uncertainty budgets, `z_m` and `w_s` can be represented by binary variables. The bilinear terms

```text
z_m * beta_m
w_s * alpha_s
```

are replaced by exact binary-continuous linearizations.

A safe bound for each `alpha_s` is

```text
max_m max(shortage_penalty_m - shipping_cost_sm, 0),
```

because larger values cannot improve dual feasibility once every `beta_m` is already bounded by its shortage penalty.

The implementation re-solves the selected worst-case scenario through the original primal recourse LP and checks primal/dual agreement on every MILP adversary call.

### Fractional uncertainty budgets

The binary MILP adversary is exact for integer budgets. For fractional budgets such as `Gamma_demand = 1.5`, `method="auto"` deliberately falls back to exact budget-polytope extreme-point enumeration.

The enumeration oracle remains in the repository as an independent verification reference rather than being removed after introducing the MILP.

## Column-and-constraint generation

The restricted master begins with the nominal scenario. Every generated uncertainty scenario receives its own recourse copy and the epigraph variable `eta` bounds scenario recourse cost.

At each iteration:

1. solve the restricted master;
2. fix the first-stage capacity decision;
3. solve the adversarial subproblem;
4. obtain a valid robust upper bound from the current first-stage decision;
5. add the worst-case scenario if it is new;
6. stop when the upper/lower bound gap is within tolerance.

The restricted-master objective is a lower bound. The adversarial evaluation of any fixed first-stage solution supplies a valid robust upper bound.

## Controlled baselines

Three policy classes are evaluated on the same synthetic instance.

### Deterministic

All decisions are optimized against nominal demand and full supplier availability.

### Static budgeted robust

Activation, capacity, shipment and shortage decisions are all **here-and-now** and shared across every uncertainty scenario. This is the non-adaptive comparator.

This repository intentionally calls it `static budgeted robust`, not a Bertsimas-Sim counterpart. The companion `robust-supply-chain-network-optimization` repository contains the algebraic Bertsimas-Sim robust counterpart. Keeping the labels separate avoids conflating two different formulations.

### Adjustable robust C&CG

Activation and capacity are first-stage decisions. Shipment and shortage are scenario-adaptive recourse decisions.

The difference between static and adjustable robust objectives is reported as the **value of adjustability** for the benchmark instance.

## Reproducible benchmark

Default configuration:

```text
Gamma_demand      = 2
Gamma_disruption  = 1
adversarial oracle = dualized MILP
```

Validated local result:

| Method | Objective | Reserved capacity |
|---|---:|---|
| Deterministic | 1098.550000 | [96.000, 0.000, 89.000] |
| Adjustable robust C&CG | 1500.928814 | [72.133, 95.000, 90.000] |
| Static budgeted robust | 2947.000000 | [105.000, 95.000, 90.000] |

For this fixture:

```text
robustness premium vs deterministic = 402.378814
value of adjustability              = 1446.071186
```

These are benchmark-specific measurements, not universal dominance claims.

The converged adjustable solution has an independently enumerated worst case

```text
demand       = [52, 44, 64, 54]
availability = [1.00, 1.00, 0.55]
shortage     = [0, 0, 0, 0]
```

and the enumeration audit reproduces the C&CG robust objective.

## C&CG convergence

For the same default configuration:

| Iteration | Master scenarios | Lower bound | Upper bound | Gap | Worst-case shortage |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1098.550000 | 2582.750000 | 1484.200000 | 73.050 |
| 2 | 2 | 1479.600000 | 1503.800000 | 24.200000 | 0.000 |
| 3 | 3 | 1500.928814 | 1500.928814 | ~0 | 0.000 |

The full uncertainty set contains 18 joint extreme scenarios for this configuration, while C&CG converges after adding only the scenarios needed by the restricted master.

## Gamma sensitivity

With `Gamma_disruption = 1` fixed:

| Gamma demand | Robust objective | Total reserved capacity | Worst-case shortage | C&CG iterations |
|---:|---:|---:|---:|---:|
| 0 | 1314.964 | 218.545 | 0.000 | 3 |
| 1 | 1408.476 | 242.441 | 0.059 | 4 |
| 2 | 1500.929 | 257.133 | 0.000 | 3 |
| 3 | 1576.529 | 271.133 | 0.000 | 3 |
| 4 | 1644.100 | 285.500 | 0.000 | 2 |

The robust objective increases with the demand protection budget on this instance. Worst-case shortage need not be monotone because the first-stage capacity decision is reoptimized at every budget value.

The CLI can generate the complete `Gamma_demand x Gamma_disruption` grid.

## Repository structure

```text
src/aro_ccg/
  data.py          validated synthetic instance
  uncertainty.py   budget-polytope extreme points
  recourse.py      recourse LP + enumeration and dualized-MILP adversaries
  master.py        restricted robust master MILP
  ccg.py           bound-tracked C&CG
  baselines.py     deterministic and static non-adaptive robust comparators
  experiments.py   Gamma sensitivity grid
  evaluation.py    independent robust audit
  __main__.py      CLI benchmark

tests/
  test_uncertainty.py
  test_recourse.py
  test_adversary.py
  test_ccg.py
  test_baselines.py
  test_experiments.py
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.10+ is supported.

## Run

Default benchmark:

```bash
python -m aro_ccg --gamma-demand 2 --gamma-disruption 1 --oracle milp
```

Fractional-budget exact fallback:

```bash
python -m aro_ccg --gamma-demand 1.5 --gamma-disruption 1 --oracle auto
```

Full sensitivity grid:

```bash
python -m aro_ccg --gamma-demand 2 --gamma-disruption 1 --oracle milp --sweep
```

## Verification contract

The regression suite checks:

- exact integer and fractional budget extreme points;
- recourse feasibility with explicit shortage;
- recourse-cost monotonicity with respect to available capacity;
- dualized adversarial MILP versus exact enumeration;
- primal/dual agreement for the adversarial MILP;
- automatic fractional-budget fallback;
- C&CG convergence;
- C&CG objective versus the full extreme-scenario master;
- deterministic <= adjustable robust <= static robust policy-class ordering on the fixture;
- independent enumeration audit of the returned first-stage decision;
- Gamma-sensitivity diagnostics and monotonic robust objective on the tested slice.

## Methodological boundary

Implemented:

- two-stage adjustable robust optimization;
- binary activation and continuous capacity reservation;
- adaptive continuous shipment/shortage recourse;
- budgeted demand and supplier-disruption uncertainty;
- dualized MILP adversarial subproblem for integer budgets;
- exact fractional-budget extreme-point oracle;
- primal/dual subproblem verification;
- bound-tracked C&CG;
- deterministic and static non-adaptive robust baselines;
- Gamma sensitivity experiments;
- full-scenario small-instance optimality oracle.

Not claimed:

- affine decision rules;
- K-adaptability;
- multistage robust optimization;
- decision-dependent uncertainty;
- distributionally robust optimization;
- industrial-scale decomposition.

Those are separate research directions rather than labels applied to functionality that is not implemented.

## Relationship to companion projects

```text
static Bertsimas-Sim robust counterpart
            |
            v
two-stage adjustable robust optimization + C&CG   <- this repository
            |
            +---- stochastic programming
            |
            +---- Wasserstein / KL distributionally robust optimization
```

The benchmark reports protection cost, value of adjustability, discovered worst-case scenarios, shortages, capacity decisions and convergence bounds separately.
