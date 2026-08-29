# Experimental Protocol

## Decision problem

Choose supplier capacity before demand is observed. After demand realizes, solve a shipment/shortage recourse problem. Compare empirical expected-value optimization with Wasserstein DRO under identical network economics.

## Data separation

Use disjoint scenario blocks for:

- development: implementation and debugging;
- validation: ambiguity-radius and model selection;
- final nominal test;
- final OOD stress test.

OOD scenarios must not be used to tune the Wasserstein radius.

## Radius selection

The ambiguity radius is a model-selection parameter, not a reporting knob. Large radii can become unnecessarily conservative; small radii may offer little protection. Select the radius on validation evidence only, then freeze it before final testing.

## Required comparisons

At minimum compare:

1. empirical expected-value capacity optimization;
2. Wasserstein DRO at several predeclared radii;
3. a deterministic mean-demand policy when useful for interpretation;
4. later, a classical robust-optimization baseline using the same network and cost structure.

## Required metrics

Report both nominal and OOD outcomes:

- mean total cost;
- p90 or CVaR total cost;
- worst observed total cost;
- first-stage reservation cost;
- mean recourse cost;
- shortage by market;
- selected capacity by supplier.

A DRO policy should not be declared superior because its ambiguity-set objective is larger or smaller. The operational comparison is out-of-sample performance and risk.

## Mathematical invariants

The implementation must satisfy:

- epsilon zero equals empirical expected recourse;
- worst-case recourse is non-decreasing in epsilon for a fixed decision;
- every capacity vector respects physical supplier limits;
- recourse explicitly accounts for unmet demand through shortage variables;
- all policies face identical final scenarios.

## Stress tests

Extend the OOD campaign to include:

- demand mean shift;
- demand variance increase;
- correlation shift between markets;
- supplier capacity loss;
- lane-cost shock;
- combined demand and supply disruption.

Report each stress family separately instead of averaging them into one score.

## Research claim discipline

The current finite-support Wasserstein model is an auditable research baseline. Claims about large-scale supply-chain DRO should wait until the network is expanded and the capacity grid is replaced by a scalable mathematical optimization or decomposition method.
