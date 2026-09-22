# Generative Supply-Chain Scenarios + Stochastic Optimization

A supply-chain uncertainty project comparing classical and learned conditional scenario generators by both **distributional fidelity** and **downstream decision quality**.

```text
planning context
   ↓
conditional demand + lead-time generator
   ↓
finite scenario set
   ↓
risk-aware first-stage reserve-capacity optimization
   ↓
independent true-distribution Monte Carlo evaluation
```

Compared scenario generators:

- nearest-neighbor conditional bootstrap;
- local Gaussian residual generator with a full joint covariance;
- independently implemented conditional VAE in PyTorch.

The project does not assume that a lower VAE reconstruction loss implies better operations-research decisions.

## Synthetic process

The hidden conditional generator contains nonlinear demand effects, seasonality, heteroskedastic serially correlated demand noise, correlated lead-time variation, and occasional demand-surge/supplier-disruption events. The learned generator never receives this hidden equation.

## Decision model

Before future paths are known, the planner chooses integer same-period expedited reserve capacity. Normal replenishment is capacity-limited and arrives after realized scenario lead time. Contracted reserve is cheaper than spot emergency supply but incurs an up-front reservation charge.

For each candidate reserve level, the supplied scenario objective is evaluated exactly:

```text
mean scenario cost + lambda * CVaR_alpha(scenario cost)
```

Every integer reserve level from zero to the declared maximum is enumerated. Thus `OPTIMAL_ENUMERATION` means exact optimality for this finite one-dimensional first-stage model, not for an arbitrary supply-chain network.

## Evaluation

Generated scenarios are compared with an independent conditional Monte Carlo sample from the hidden synthetic process.

Distribution diagnostics include demand/lead-time mean, standard deviation, lag-1 dependence, cross-variable correlation and tail quantiles.

Decision diagnostics include:

- out-of-sample mean cost;
- out-of-sample CVaR;
- protected fill rate;
- stockout-period rate;
- chosen reserve capacity.

## Development run

Seed-42 development run:

```text
training paths       500
validation paths     140
held-out contexts      8
CVAE epochs            18
generated scenarios    80/context
evaluation paths      160/context
```

Result:

```text
method                    moment distance   mean cost      CVaR      fill   stockout   reserve
Conditional bootstrap          0.1419        1930.75     2508.72   0.9854   0.1036     68.75
Gaussian residual              0.1422        1930.87     2508.55   0.9854   0.1027     68.88
Conditional VAE                0.2125        1927.73     2509.90   0.9847   0.1086     65.75
```

The CVAE did **not** beat the classical generators on the declared moment-distance diagnostic. It selected slightly less reserve capacity and obtained slightly lower mean cost in this finite run, while CVaR and service were similar/slightly worse. No generative-model superiority claim is made.

This negative/ambiguous result is kept deliberately: the point is to evaluate scenario generators by the decisions they induce, not to advertise a learned model regardless of evidence.

## GitHub Actions validation

A real GitHub-hosted Ubuntu runner validated the end-to-end pipeline with:

```text
Python          3.12.14
PyTorch         2.13.0+cpu
NumPy           2.5.2
SciPy           1.18.1
scikit-learn    1.9.0
```

The remote regression suite passed all **6/6 tests**.

The CI smoke experiment used:

```text
horizon                 5
training paths         160
validation paths        50
held-out contexts        3
CVAE epochs              5
generated scenarios     32/context
evaluation paths        64/context
```

Runner-observed result:

```text
best validation reconstruction MSE: 1.03573

method                    moment distance   mean cost      CVaR      fill   stockout   reserve
Conditional bootstrap          0.2000        1564.45     1995.78   0.9806   0.1167     69.00
Gaussian residual              0.2075        1562.95     1994.28   0.9806   0.1167     68.00
Conditional VAE                0.3690        1579.67     2019.54   0.9681   0.1552     61.67
```

In this short CI training run the CVAE did **not** outperform either classical generator on the declared moment-distance metric, expected cost, CVaR, fill rate, or stockout-period rate. This is treated as a validated negative learned-model result rather than as a speedup or quality claim. The smoke run exists to validate the complete training → scenario generation → stochastic decision → independent evaluation path.

## Regression tests

The suite checks:

- deterministic synthetic path generation;
- valid nonnegative demand and positive lead times;
- bootstrap/Gaussian scenario shapes;
- actual CVAE training and sampling;
- exact finite reserve-enumeration objective against an independent brute-force reconstruction;
- finite operational evaluation.

## Run

```bash
pip install -r requirements.txt
python run_generative_supply_chain.py --self-test
python -m unittest discover -s tests -v
python run_generative_supply_chain.py
```

## Scope

Exact claim: reserve enumeration is exact only for the declared finite integer first-stage model and supplied scenario sample.

Not claimed:

- the CVAE dominates bootstrap/Gaussian scenario generation;
- the synthetic process represents a real firm;
- the finite scenario solution is the true stochastic optimum;
- the reported moment distance is a universal generative-quality metric;
- the model is a production-ready digital twin.
