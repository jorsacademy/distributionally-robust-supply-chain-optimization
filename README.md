# Distributionally Robust Supply Chain Optimization

Research-oriented Industrial Engineering / Operations Research project for supply-chain decisions under **distribution shift**. The initial model is a data-driven single-period ordering problem evaluated under an empirical Wasserstein ambiguity set; the repository is designed to expand toward multi-echelon and two-stage network models.

## Research question

How much nominal performance should an operator sacrifice to obtain decisions that remain reliable when the future demand distribution differs from the empirical training sample?

## Controllers compared

- empirical expected-value optimization;
- scenario stress-test optimization;
- Wasserstein distributionally robust optimization (DRO);
- simple safety-stock/base-stock policies for interpretability.

## Wasserstein DRO core

For a candidate decision, the inner adversary redistributes empirical probability mass across a finite demand support. A transport plan is constrained by a Wasserstein transportation budget. The adversary maximizes operating cost; the outer optimization selects the decision minimizing this worst-case expectation.

The included implementation solves the inner transport LP exactly with `scipy.optimize.linprog` and searches the one-dimensional order decision on an explicit grid. This keeps the mathematics auditable before extending to larger mixed-integer network models.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m supply_dro.experiment
```

## Repository map

```text
src/supply_dro/
  model.py          # cost model and empirical demand data
  wasserstein.py    # worst-case transport LP
  optimize.py       # nominal and DRO decisions
  experiment.py     # nominal/OOD stress campaign
tests/
  test_dro.py
configs/
  experiment.json
.github/workflows/
  ci.yml
```

## Evaluation contract

Do not select a model on nominal expected cost alone. Report:

- nominal expected cost;
- worst-case ambiguity-set cost;
- CVaR / tail cost on independent scenarios;
- stockout frequency and service level;
- sensitivity to ambiguity radius;
- stability under demand mean/variance shifts.

## Research extensions

1. two-stage capacitated supplier allocation;
2. lead-time and supplier-disruption uncertainty;
3. multi-echelon inventory;
4. mixed-integer facility/capacity decisions;
5. comparison with robust optimization and classical stochastic programming;
6. decision-focused calibration of ambiguity radius.

## License

MIT
