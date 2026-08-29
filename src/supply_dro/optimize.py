import numpy as np

from .wasserstein import worst_case_expected_cost


def candidate_grid(empirical, margin=20):
    lo = max(0, int(np.floor(np.min(empirical))) - margin)
    hi = int(np.ceil(np.max(empirical))) + margin
    return np.arange(lo, hi + 1, dtype=float)


def optimize_nominal(empirical, model):
    grid = candidate_grid(empirical)
    scored = [(float(np.mean(model.cost(q, empirical))), q) for q in grid]
    cost, q = min(scored)
    return {"order_quantity": float(q), "objective": float(cost)}


def optimize_dro(empirical, model, epsilon):
    grid = candidate_grid(empirical)
    best = None
    for q in grid:
        inner = worst_case_expected_cost(q, empirical, model, epsilon)
        candidate = (inner["worst_case_cost"], q, inner)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return {
        "order_quantity": float(best[1]),
        "objective": float(best[0]),
        "adversary": best[2],
    }


def evaluate(order_quantity, demand, model, alpha=0.95):
    costs = np.asarray(model.cost(order_quantity, demand), dtype=float)
    threshold = np.quantile(costs, alpha)
    tail = costs[costs >= threshold]
    service_level = float(np.mean(np.asarray(demand) <= order_quantity))
    return {
        "mean_cost": float(np.mean(costs)),
        "cvar": float(np.mean(tail)),
        "service_level": service_level,
        "stockout_rate": 1.0 - service_level,
    }
