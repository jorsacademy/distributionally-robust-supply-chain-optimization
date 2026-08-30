# Final Report — Distributionally Robust Supply Chain Optimization

## Scope

This repository now contains two deliberately separated but compatible research layers:

1. a frozen two-stage supply-network uncertainty benchmark;
2. a finite-horizon sequential capacity-control benchmark using the same LP recourse model.

The implemented stack is considered feature-complete:

1. nominal empirical stochastic optimization;
2. two-stage capacity reservation with LP recourse;
3. finite-support demand-only Wasserstein DRO;
4. finite-set classical robust optimization over named demand/disruption stresses;
5. joint demand-and-availability Wasserstein ambiguity;
6. validation-only ambiguity-radius calibration;
7. untouched nominal, demand-shift and joint-shift final blocks;
8. paired bootstrap comparisons against the nominal policy;
9. sequential capacity adjustment under observable normal/stress signals;
10. exact finite-MDP dynamic programming;
11. tabular Q-learning from sampled transitions;
12. static nominal, static robust and myopic OR control baselines;
13. multi-seed RL aggregation, common random numbers, latency and paired inference;
14. reproducible configuration, tests and CI.

## Two-stage model

First-stage decisions reserve supplier capacity `x_s`. After demand `d_m` and supplier availability `a_s` are realized, the recourse problem chooses shipment `y_sm` and shortage `u_m`:

```text
min  sum(c_sm y_sm) + sum(p_m u_m)
s.t. sum_m y_sm <= a_s x_s                  for each supplier s
     sum_s y_sm + u_m >= d_m                 for each market m
     y_sm, u_m >= 0
```

The first-stage objective adds reservation cost to the selected uncertainty treatment of recourse cost.

## Static decision paradigms

### Nominal stochastic

Minimizes reservation cost plus empirical mean recourse under the training demand sample.

### Demand-only Wasserstein DRO

Reweights the empirical vector-demand distribution inside a finite-support 1-Wasserstein ball. Supplier availability remains fixed at full availability.

### Joint demand/disruption Wasserstein DRO

The uncertainty state contains both demand and supplier availability. The transport metric combines normalized L1 demand distance with a separately weighted availability distance.

### Classical robust

Minimizes the maximum total cost over a finite named stress set containing demand surges, individual supplier disruptions and a joint stress case.

These models are intentionally not treated as interchangeable forms of robustness.

## Radius calibration

Wasserstein radii are selected using validation data only. The selection criterion is validation p90 total cost. Candidate radii are frozen in `configs/final_evaluation.json`.

The final evaluation blocks are not used for radius selection, capacity-grid selection or model redesign.

## Static frozen final blocks

The static final campaign evaluates every policy on the same paired realizations in:

- `nominal_final`;
- `demand_shift`;
- `joint_shift`.

For each policy and block, the benchmark reports mean, p90 and worst total cost. Scenario-level paired differences against the nominal policy are summarized with a deterministic bootstrap confidence interval and win rate.

## Sequential control extension

The RL/control extension is a separate finite-horizon MDP. It does not retroactively redefine the two-stage DRO problem.

### State

The controller observes:

- current reserved-capacity levels for both suppliers on a discrete grid;
- a public Markov regime signal: `normal` or `stress`.

### Action

The controller chooses the next supplier-capacity vector. Each supplier may move by at most one capacity-grid step per period.

### Random transition

Conditional on the observed regime, demand and supplier availability are sampled from a finite joint operational distribution. The same LP recourse model computes shipment and shortage cost. The regime then follows a two-state Markov transition matrix.

### Stage cost

```text
reservation cost
+ adjustment cost
+ LP recourse cost
```

### Exact dynamic-programming oracle

Because the benchmark has a small finite state/action space and known transition model, backward dynamic programming provides the exact optimal policy for the discretized MDP. This is the principal sequential-control reference.

### Q-learning

Tabular Q-learning is trained from sampled transitions only. It does not receive DP values, optimal DP actions or future random outcomes.

Three independent Q-learning seeds are trained. At evaluation time their costs are first averaged within each common environment seed; model seeds are therefore not treated as extra independent test observations.

### Sequential baselines

The final control comparison includes:

- exact dynamic DP;
- one-step myopic OR;
- static nominal target control;
- static classical-robust target control;
- tabular Q-learning.

This comparison is intentionally demanding: Q-learning must earn its place against a strong one-step optimization controller and an exact finite-MDP oracle.

## Common-random-number evaluation

Every sequential controller is rolled out with the same environment seeds. Exogenous demand, availability and signal-transition random numbers depend on the environment seed rather than the chosen action. Paired total-cost comparisons therefore use common random numbers.

The control report includes:

- mean total cost;
- cost gap to exact DP;
- paired bootstrap 95% interval;
- paired win rate;
- decision latency.

## Validation invariants

A result is accepted only if:

1. Wasserstein radius zero reproduces empirical expected recourse;
2. joint radius zero reproduces empirical expected recourse under joint demand/availability states;
3. supplier availability remains in `[0, 1]`;
4. all unmet demand is represented explicitly by shortage recourse;
5. first-stage reservation cost is included in every static final objective;
6. candidate radii are chosen only from validation data;
7. all static policies are evaluated on identical final realizations;
8. nominal, demand-DRO, joint-DRO and classical-robust results are reported separately;
9. sequential actions respect the discrete capacity grid and adjustment bound;
10. Q-learning receives sampled transition feedback only;
11. exact DP is an evaluation/control oracle rather than a training label source;
12. RL model seeds are aggregated within environment seeds before paired inference;
13. negative/null results are retained.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m supply_dro.experiment
python -m supply_dro.final_campaign
python -m supply_dro.rl_experiment
```

## Interpretation

The static research question is not “does DRO always win?” A conservative policy may increase nominal cost while reducing tail or disruption exposure. Such a trade-off is acceptable only when the resilience gain is measurable on untouched scenarios.

The sequential research question is likewise not “does RL beat OR?” Exact dynamic programming should dominate the discretized MDP when its assumptions are correct. The meaningful comparison is whether sampled-transition Q-learning approaches that reference efficiently and whether it adds enough adaptivity relative to static or myopic controls to justify its learning complexity.

A result in which myopic OR or a static robust target outperforms Q-learning is a valid result and must remain visible.

## Scope boundary

The repository is complete as a small uncertainty-and-control benchmark. Multi-echelon networks, facility opening, integer expansion, continuous-action RL, endogenous disruptions, travel/lead-time state and large-network decomposition are separate research extensions rather than further additions to this frozen repository.
