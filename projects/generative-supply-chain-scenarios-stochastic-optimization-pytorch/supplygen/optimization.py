from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PlanningParameters:
    reservation_cost: float = 1.5
    regular_cost: float = 4.0
    reserved_usage_cost: float = 5.8
    emergency_cost: float = 11.5
    holding_cost: float = 0.6
    shortage_cost: float = 13.0
    base_regular_capacity: float = 68.0
    max_reserved_capacity: float = 70.0
    initial_inventory: float = 8.0
    alpha: float = 0.90
    cvar_weight: float = 0.20

@dataclass(frozen=True)
class CapacityPlan:
    reserved_capacity: float
    objective: float
    status: str
    scenario_count: int

def _simulate_given_reserve(reserve, demand, lead, p: PlanningParameters):
    d=np.asarray(demand,float); l=np.asarray(lead,int); H=len(d)
    arrivals=np.zeros(H+int(np.max(l))+2,float)
    inv=float(p.initial_inventory)
    total=p.reservation_cost*reserve
    native_filled=0.0
    protected_filled=0.0
    stockout_periods=0

    for t in range(H):
        inv += arrivals[t]
        regular=min(p.base_regular_capacity, max(d[t]-max(inv,0.0),0.0))
        arr_t=t+int(l[t])
        if arr_t < len(arrivals):
            arrivals[arr_t]+=regular
        total += p.regular_cost*regular

        native=min(max(inv,0.0),d[t])
        remaining=d[t]-native
        reserved_use=min(reserve,remaining)
        remaining-=reserved_use
        total += p.reserved_usage_cost*reserved_use

        emergency=remaining
        if emergency>1e-12:
            stockout_periods += 1
            total += p.emergency_cost*emergency

        native_filled += native
        protected_filled += native + reserved_use
        inv=max(inv-d[t],0.0)
        total += p.holding_cost*inv

    protected_fill=protected_filled/max(float(np.sum(d)),1e-12)
    stockout_rate=stockout_periods/H
    return float(total), float(protected_fill), float(stockout_rate)

def solve_capacity_saa(scenarios_demand, scenarios_lead, p: PlanningParameters):
    D=np.asarray(scenarios_demand,float)
    L=np.asarray(scenarios_lead,int)
    if D.ndim!=2 or L.shape!=D.shape:
        raise ValueError("scenario shapes mismatch")
    candidates=np.arange(0,int(round(p.max_reserved_capacity))+1,dtype=float)
    best=None
    for r in candidates:
        costs=np.asarray([_simulate_given_reserve(r,d,l,p)[0] for d,l in zip(D,L)],float)
        var=np.quantile(costs,p.alpha,method="higher")
        tail=costs[costs>=var-1e-12]
        cvar=float(tail.mean()) if len(tail) else float(var)
        obj=float(costs.mean() + p.cvar_weight*cvar)
        if best is None or obj < best[0]-1e-12:
            best=(obj,r)
    return CapacityPlan(float(best[1]),float(best[0]),"OPTIMAL_ENUMERATION",len(D))

def evaluate_plan(plan: CapacityPlan, demand, lead, p: PlanningParameters):
    return _simulate_given_reserve(plan.reserved_capacity,demand,lead,p)
