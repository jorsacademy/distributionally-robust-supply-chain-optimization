import numpy as np
from scipy.optimize import linprog


def worst_case_expected_cost(order_quantity, empirical, model, epsilon):
    """Solve the inner Wasserstein adversary on the empirical finite support.

    Source mass is fixed at the empirical distribution. The adversary transports
    that mass across observed demand support points with total transportation
    cost bounded by epsilon, maximizing the operating cost of the fixed order.
    """
    empirical = np.asarray(empirical, dtype=float)
    support, counts = np.unique(empirical, return_counts=True)
    source_prob = counts / counts.sum()
    n = len(support)

    destination_cost = model.cost(order_quantity, support)
    objective = -np.tile(destination_cost, n)

    A_eq = np.zeros((n, n * n))
    for i in range(n):
        A_eq[i, i * n:(i + 1) * n] = 1.0
    b_eq = source_prob

    transport = np.abs(support[:, None] - support[None, :]).reshape(1, -1)
    result = linprog(
        objective,
        A_ub=transport,
        b_ub=np.array([float(epsilon)]),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * (n * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)

    plan = result.x.reshape(n, n)
    adversarial_prob = plan.sum(axis=0)
    return {
        "worst_case_cost": float(-result.fun),
        "support": support,
        "adversarial_prob": adversarial_prob,
        "transport_used": float(np.sum(plan * np.abs(support[:, None] - support[None, :]))),
    }
