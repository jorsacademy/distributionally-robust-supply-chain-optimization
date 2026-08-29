from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedBootstrap:
    mean_difference: float
    ci_low: float
    ci_high: float
    win_rate: float
    n: int


def paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    n_bootstrap: int = 3000,
    seed: int = 123,
) -> PairedBootstrap:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired arrays must be equal-length one-dimensional arrays")
    if len(candidate) == 0:
        raise ValueError("at least one paired observation is required")
    difference = candidate - reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))
    means = difference[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return PairedBootstrap(
        mean_difference=float(np.mean(difference)),
        ci_low=float(low),
        ci_high=float(high),
        win_rate=float(np.mean(candidate < reference)),
        n=len(candidate),
    )
