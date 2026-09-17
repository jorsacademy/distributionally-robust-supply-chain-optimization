# Research Series — Stochastic, Robust, and Risk-Aware Optimization

This repository belongs to a broader uncertainty-aware optimization series. The projects stay separate because they encode different uncertainty assumptions and therefore answer different decision questions.

## Core uncertainty sequence

1. **two-stage-stochastic-capacity-planning** — foundational two-stage stochastic programming.
2. **benders-decomposition-capacity-planning** — the same broad two-stage recourse idea studied through decomposition rather than only extensive-form optimization.
3. **stochastic-cvrp-sample-average-approximation-python** — Sample Average Approximation in stochastic routing.
4. **parallel-monte-carlo-stochastic-optimization-python** — simulation / Monte Carlo evaluation for stochastic optimization.
5. **chance-constrained-inventory-optimization-python** — probabilistic service/feasibility constraints.
6. **robust-supply-chain-network-optimization** — classical robust optimization with explicit uncertainty sets.
7. **robust-healthcare-inventory-optimization** — robust inventory decisions in a healthcare setting.
8. **wasserstein-dro-inventory-optimization-python** — Wasserstein distributionally robust optimization for inventory.
9. **distributionally-robust-supply-chain-optimization** — supply-network DRO under demand shift and disruptions, extended with sequential control.
10. **distributionally-robust-decision-focused-learning** — DRO combined with end-to-end decision-focused learning.
11. **conformal-prediction-robust-inventory-optimization-python** — predictive uncertainty sets calibrated using conformal prediction and then passed to optimization.
12. **generative-supply-chain-scenarios-stochastic-optimization-pytorch** — learned scenario generation feeding stochastic optimization.
13. **risk-based-resource-allocation-milp-python** and **risk-aware-convoy-escort-allocation-milp** — explicit risk-aware objective/constraint formulations.
14. **sddp-multistage-energy-storage** and **mpi-sppy-multistage-stochastic-planning** — multistage stochastic decision making.

## Conceptual differences

- **Stochastic programming:** assumes a probability model and optimizes expected or risk-adjusted performance.
- **SAA:** replaces expectations with a sampled finite scenario set.
- **Chance constraints:** control the probability of constraint violation.
- **Robust optimization:** protects against every realization in a specified uncertainty set.
- **DRO:** protects against distributions in an ambiguity set rather than individual realizations.
- **Conformal optimization:** derives finite-sample predictive uncertainty sets before optimization.
- **Generative scenarios:** learns a scenario generator, but the downstream optimizer remains a separate decision layer.
- **Multistage stochastic optimization:** decisions adapt repeatedly as information arrives.

These distinctions are methodological, not cosmetic. The repositories should therefore be read as a progression across uncertainty models rather than merged into one codebase.

## Closely related decision-learning projects

`predict-then-optimize-production-planning-spo-plus-pytorch`, `decision-focused-learning-spo`, `contextual-optimization-newsvendor`, and `differentiable-black-box-supplier-selection-pytorch` are adjacent but belong primarily to the predict-then-optimize / decision-focused-learning series.
