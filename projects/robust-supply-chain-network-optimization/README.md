# Robust Supply Chain Network Optimization

A reproducible Operations Research case study for capacitated supply-chain network design under demand uncertainty. The project compares a nominal facility-location MILP with a Bertsimas-Sim budgeted robust counterpart and then stress-tests the resulting designs on demand scenarios that were not used by the optimizer.

The implementation uses `scipy.optimize.milp`, which solves the mixed-integer linear programs with the HiGHS backend distributed with SciPy.

## Problem

A company must decide which distribution facilities to open and assign every customer zone to exactly one open facility. Customer demand is uncertain.

For facility `i` and customer `j`:

- `y_i` is 1 when facility `i` is open,
- `x_ij` is 1 when customer `j` is assigned to facility `i`,
- `dbar_j` is nominal demand,
- `dhat_j` is the maximum positive demand deviation,
- `K_i` is facility capacity,
- `Gamma` is the budget of uncertainty.

The nominal design problem contains

```text
sum_i x_ij = 1                              for every customer j
x_ij <= y_i                                 for every i, j
sum_j dbar_j x_ij <= K_i y_i                for every facility i
```

and minimizes fixed opening cost plus nominal shipping cost.

## Bertsimas-Sim robust capacity

Demand is represented as

```text
d_j = dbar_j + dhat_j u_j
0 <= u_j <= 1
sum_j u_j <= Gamma
```

inside each facility-capacity constraint. `Gamma` controls how many demand coefficients may move toward their declared worst-case deviations simultaneously. It is a protection budget, not a probability and not a scenario count.

For every facility, the nonlinear worst-case capacity expression can be replaced by the linear robust counterpart

```text
sum_j dbar_j x_ij + Gamma z_i + sum_j p_ij <= K_i y_i
z_i + p_ij >= dhat_j x_ij
z_i >= 0
p_ij >= 0
```

The resulting model is still a MILP. This is the central idea from Bertsimas and Sim's budgeted-uncertainty construction: conservatism can be increased gradually rather than requiring every uncertain coefficient to take its worst value at once.

Special cases:

- `Gamma = 0` reproduces the nominal capacity model.
- `Gamma = 1` protects each facility against the largest assigned demand deviation.
- fractional values such as `Gamma = 2.5` are valid.
- sufficiently large `Gamma` protects against all declared deviations of the customers assigned to a facility.

## Why this is different from stochastic programming

This repository implements a static robust model. Facility openings and customer assignments are here-and-now decisions, and there is no recourse after demand is observed. The uncertainty set is deterministic and no demand probability distribution is required by the optimizer.

The stress-test distribution in `evaluation.py` is used only to compare designs after optimization. It does not turn the robust formulation into a stochastic program and it does not imply a formal chance-constraint guarantee.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_gamma_sweep.py
├── src/robust_scn/
│   ├── __init__.py
│   ├── __main__.py
│   ├── data.py
│   ├── evaluation.py
│   ├── experiment.py
│   └── model.py
├── tests/
│   ├── test_budget.py
│   ├── test_evaluation.py
│   └── test_model.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.10+ is supported.

## Run a robustness sweep

```bash
python -m robust_scn \
  --gammas 0 1 2 3 4 6 \
  --scenarios 5000 \
  --seed 2026 \
  --shock-probability 0.35
```

or

```bash
python examples/run_gamma_sweep.py
```

The output reports, for every `Gamma`:

- optimized nominal objective value,
- open facilities,
- customer-to-facility assignments,
- minimum robust capacity slack,
- out-of-sample scenario violation rate,
- mean and 95th-percentile capacity excess in the stress test.

The objective is intentionally based on nominal shipping costs. Therefore the increase in objective value as `Gamma` rises is directly interpretable as a price paid for additional capacity protection in this model.

## Reproducible demo behavior

The synthetic instance is chosen so that the nominal solution can exploit capacity aggressively. With the default data, `Gamma = 0` opens three facilities. Increasing the protection budget eventually makes a four-facility design preferable. The exact customer assignments are solver outputs rather than hard-coded rules.

The evaluation layer generates bounded positive demand shocks from a separate random-number generator. A robust solution is never declared "better" merely because its optimization objective is larger or smaller; the repository reports both design cost and realized capacity-violation behavior.

## Tests

```bash
python -m pytest
```

The suite verifies:

- the closed-form budgeted worst-case calculation for integer and fractional `Gamma`,
- single-source assignment and nominal feasibility,
- analytical robust-capacity feasibility of solved designs,
- nonnegative price of robustness on the demo instance,
- a small instance with a known structural requirement,
- reproducible and bounded stress scenarios,
- conservation of total demand when facility loads are computed,
- validity of reported violation probabilities.

GitHub Actions runs compile and test jobs on Python 3.10 and 3.12.

## Methodological notes

1. The robust model protects capacity constraints against the stated uncertainty set. It does not guarantee zero violations for arbitrary demand realizations outside that set.
2. `Gamma` is applied separately to each facility-capacity constraint, which matches the standard row-wise budgeted-uncertainty construction.
3. The model uses single sourcing. Allowing demand splitting would change the decision structure and can materially change the price of robustness.
4. The uncertainty is one-sided because the operational concern is demand exceeding nominal values. Lower demand does not threaten capacity feasibility in this model.
5. The stress-test distribution is deliberately synthetic. A production implementation should estimate demand ranges and validation scenarios from data and should examine model misspecification explicitly.

## References

- D. Bertsimas and M. Sim, *The Price of Robustness*, Operations Research 52(1), 35-53, 2004. https://doi.org/10.1287/opre.1030.0065
- SciPy documentation, `scipy.optimize.milp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
- HiGHS optimization software: https://highs.dev/

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
