# Final Report — Distributionally Robust Supply Chain Optimization

## Scope

This repository is a compact research benchmark for two-stage supply-network decisions under demand distribution shift and supplier-capacity disruption.

The implemented stack is now considered feature-complete for this benchmark:

1. nominal empirical stochastic optimization;
2. two-stage capacity reservation with LP recourse;
3. finite-support demand-only Wasserstein DRO;
4. finite-set classical robust optimization over named demand/disruption stresses;
5. joint demand-and-availability Wasserstein ambiguity;
6. validation-only ambiguity-radius calibration;
7. untouched nominal, demand-shift and joint-shift final blocks;
8. paired bootstrap comparisons against the nominal policy;
9. reproducible configuration, tests and CI.

## Two-stage model

First-stage decisions reserve supplier capacity `x_s`. After demand `d_m` and supplier availability `a_s` are realized, the recourse problem chooses shipment `y_sm` and shortage `u_m`:

```text
min  sum(c_sm y_sm) + sum(p_m u_m)
s.t. sum_m y_sm <= a_s x_s                  for each supplier s
     sum_s y_sm + u_m >= d_m                 for each market m
     y_sm, u_m >= 0
```

The first-stage objective adds reservation cost to the selected uncertainty treatment of recourse cost.

## Four decision paradigms

### Nominal stochastic

Minimizes reservation cost plus empirical mean recourse under the training demand sample.

### Demand-only Wasserstein DRO

Reweights the empirical vector-demand distribution inside a finite-support 1-Wasserstein ball. Supplier availability remains fixed at full availability.

### Joint demand/disruption Wasserstein DRO

The uncertainty state contains both demand and supplier availability. The transport metric combines normalized L1 demand distance with a separately weighted availability distance. This allows probability mass to move toward states that are simultaneously demand-intensive and capacity-impaired.

### Classical robust

Minimizes the maximum total cost over a finite named stress set containing demand surges, individual supplier disruptions and a joint stress case.

These models are intentionally not treated as interchangeable forms of “robustness.” Their ambiguity assumptions differ materially.

## Radius calibration

Wasserstein radii are selected using validation data only. The selection criterion is validation p90 total cost. Candidate radii are frozen in `configs/final_evaluation.json`.

The final evaluation blocks are not used for radius selection, capacity-grid selection or model redesign. This protects the final comparison from test leakage.

## Frozen final blocks

The final campaign evaluates every policy on the same paired realizations in three blocks:

- `nominal_final`: demand from the nominal generator, full supplier availability;
- `demand_shift`: higher-demand distribution, full supplier availability;
- `joint_shift`: shifted demand plus random supplier-availability losses.

For each policy and block, the benchmark reports mean, p90 and worst total cost. Scenario-level paired differences against the nominal policy are summarized with a deterministic bootstrap confidence interval and win rate.

## Validation invariants

A result is accepted only if:

1. Wasserstein radius zero reproduces empirical expected recourse;
2. joint radius zero reproduces empirical expected recourse under joint demand/availability states;
3. supplier availability remains in `[0, 1]`;
4. all unmet demand is represented explicitly by shortage recourse;
5. first-stage reservation cost is included in every final objective;
6. candidate radii are chosen only from validation data;
7. all policies are evaluated on identical final realizations;
8. nominal, demand-DRO, joint-DRO and classical-robust results are reported separately;
9. a robust policy is not declared superior from one metric alone.

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

## Interpretation

The research question is not “does DRO always win?” A more conservative policy may increase nominal cost while reducing tail or disruption exposure. Such a trade-off is acceptable only when the resilience gain is measurable on untouched scenarios.

If the nominal solution remains best under both nominal and shifted conditions, that is a valid negative result. If classical robust dominates only named disruption stresses but performs poorly under probabilistic demand shift, that distinction is itself an important result. Likewise, joint DRO must earn its additional modeling complexity relative to demand-only DRO.

## Scope boundary

The repository is complete as a small two-stage uncertainty benchmark. Multi-echelon networks, facility opening, integer capacity expansion, lead-time dynamics, endogenous disruptions and decomposition for large networks should be separate research extensions rather than silent changes to this frozen benchmark.
