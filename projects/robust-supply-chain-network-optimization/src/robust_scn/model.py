from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .data import SupplyChainInstance


@dataclass(frozen=True)
class NetworkSolution:
    gamma: float
    objective: float
    open_facilities: tuple[int, ...]
    assignment: tuple[int, ...]
    nominal_load: np.ndarray
    robust_load: np.ndarray
    capacity_slack: np.ndarray
    mip_status: int
    mip_message: str


def budgeted_worst_case_extra(values: np.ndarray, gamma: float) -> float:
    """Return the Bertsimas-Sim budgeted worst-case sum of nonnegative deviations."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if np.any(v < -1e-12):
        raise ValueError("values must be nonnegative")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if v.size == 0 or gamma == 0:
        return 0.0

    capped_gamma = min(float(gamma), float(v.size))
    ordered = np.sort(v)[::-1]
    whole = int(math.floor(capped_gamma))
    fraction = capped_gamma - whole
    total = float(np.sum(ordered[:whole]))
    if whole < v.size:
        total += fraction * float(ordered[whole])
    return total


def _indices(n_facilities: int, n_customers: int) -> tuple[slice, slice, slice, slice]:
    y_start = 0
    y_stop = n_facilities
    x_start = y_stop
    x_stop = x_start + n_facilities * n_customers
    z_start = x_stop
    z_stop = z_start + n_facilities
    p_start = z_stop
    p_stop = p_start + n_facilities * n_customers
    return (
        slice(y_start, y_stop),
        slice(x_start, x_stop),
        slice(z_start, z_stop),
        slice(p_start, p_stop),
    )


def solve_network(
    instance: SupplyChainInstance,
    gamma: float = 0.0,
    time_limit: float | None = 30.0,
    mip_rel_gap: float = 0.0,
) -> NetworkSolution:
    """
    Solve a single-source capacitated facility-location model with budgeted demand uncertainty.

    Each customer is assigned to one facility before demand is observed. Demand for customer j is
    d_j = dbar_j + dhat_j * u_j, where 0 <= u_j <= 1. For each facility-capacity constraint,
    sum_j u_j <= gamma. The resulting Bertsimas-Sim robust counterpart remains a MILP.
    """
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be nonnegative")

    f = instance.n_facilities
    c = instance.n_customers
    y_slice, x_slice, z_slice, p_slice = _indices(f, c)
    n_vars = p_slice.stop

    def x_idx(i: int, j: int) -> int:
        return x_slice.start + i * c + j

    def p_idx(i: int, j: int) -> int:
        return p_slice.start + i * c + j

    objective = np.zeros(n_vars, dtype=float)
    objective[y_slice] = instance.fixed_cost
    for i in range(f):
        for j in range(c):
            objective[x_idx(i, j)] = (
                instance.nominal_demand[j] * instance.shipping_cost[i, j]
            )

    integrality = np.zeros(n_vars, dtype=int)
    integrality[y_slice] = 1
    integrality[x_slice] = 1

    lower = np.zeros(n_vars, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)
    upper[y_slice] = 1.0
    upper[x_slice] = 1.0

    # Rows: customer assignment, linking, robust capacity, robust dualization.
    n_rows = c + f * c + f + f * c
    A = lil_matrix((n_rows, n_vars), dtype=float)
    lb = np.full(n_rows, -np.inf, dtype=float)
    ub = np.full(n_rows, np.inf, dtype=float)
    row = 0

    # Every customer is single-sourced.
    for j in range(c):
        for i in range(f):
            A[row, x_idx(i, j)] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0
        row += 1

    # A customer can only be assigned to an open facility.
    for i in range(f):
        for j in range(c):
            A[row, x_idx(i, j)] = 1.0
            A[row, y_slice.start + i] = -1.0
            ub[row] = 0.0
            row += 1

    # Robust facility capacities:
    # sum_j dbar_j x_ij + gamma*z_i + sum_j p_ij <= capacity_i*y_i
    for i in range(f):
        A[row, y_slice.start + i] = -instance.capacity[i]
        A[row, z_slice.start + i] = float(gamma)
        for j in range(c):
            A[row, x_idx(i, j)] = instance.nominal_demand[j]
            A[row, p_idx(i, j)] = 1.0
        ub[row] = 0.0
        row += 1

    # z_i + p_ij >= dhat_j*x_ij
    for i in range(f):
        for j in range(c):
            A[row, x_idx(i, j)] = instance.demand_deviation[j]
            A[row, z_slice.start + i] = -1.0
            A[row, p_idx(i, j)] = -1.0
            ub[row] = 0.0
            row += 1

    assert row == n_rows

    options: dict[str, float | bool] = {"disp": False, "mip_rel_gap": float(mip_rel_gap)}
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive when provided")
        options["time_limit"] = float(time_limit)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options=options,
    )
    if result.x is None or result.fun is None:
        raise RuntimeError(f"MILP did not return a solution: {result.message}")
    if result.status not in (0, 1):
        raise RuntimeError(f"MILP failed with status {result.status}: {result.message}")

    vector = np.asarray(result.x)
    y = vector[y_slice]
    x = vector[x_slice].reshape(f, c)
    open_facilities = tuple(int(i) for i in np.flatnonzero(y > 0.5))
    assignment = tuple(int(np.argmax(x[:, j])) for j in range(c))

    nominal_load = np.zeros(f, dtype=float)
    robust_load = np.zeros(f, dtype=float)
    for i in range(f):
        assigned = np.array([assignment[j] == i for j in range(c)], dtype=bool)
        nominal_load[i] = float(np.sum(instance.nominal_demand[assigned]))
        deviations = instance.demand_deviation[assigned]
        robust_load[i] = nominal_load[i] + budgeted_worst_case_extra(deviations, gamma)

    capacity_slack = instance.capacity - robust_load

    return NetworkSolution(
        gamma=float(gamma),
        objective=float(result.fun),
        open_facilities=open_facilities,
        assignment=assignment,
        nominal_load=nominal_load,
        robust_load=robust_load,
        capacity_slack=capacity_slack,
        mip_status=int(result.status),
        mip_message=str(result.message),
    )
