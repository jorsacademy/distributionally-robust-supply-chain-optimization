from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .data import sample_conditional_paths
from .generators import conditional_bootstrap, gaussian_residual_generator, cvae_scenarios
from .metrics import moment_distance, cvar
from .optimization import PlanningParameters, solve_capacity_saa, evaluate_plan

@dataclass(frozen=True)
class EvaluationRow:
    method: str
    moment_distance: float
    mean_cost: float
    cvar_cost: float
    mean_fill_rate: float
    mean_stockout_rate: float
    mean_reserved_capacity: float

def evaluate_generators(train, cvae_result, contexts, *, n_scenarios=128, eval_scenarios=256, seed=42, device="cpu", params=None):
    p=params or PlanningParameters()
    records={k:[] for k in ["Conditional bootstrap","Gaussian residual","Conditional VAE"]}
    md={k:[] for k in records}

    for i,ctx in enumerate(np.asarray(contexts,float)):
        gens={
            "Conditional bootstrap":conditional_bootstrap(train,ctx,n_scenarios=n_scenarios,seed=seed+17*i),
            "Gaussian residual":gaussian_residual_generator(train,ctx,n_scenarios=n_scenarios,seed=seed+31*i),
            "Conditional VAE":cvae_scenarios(cvae_result,ctx,n_scenarios=n_scenarios,seed=seed+43*i,device=device),
        }
        rep=np.repeat(ctx[None,:],eval_scenarios,axis=0)
        true_d,true_l=sample_conditional_paths(rep,seed=seed+1_000_003+101*i,horizon=train.horizon)

        for name,sc in gens.items():
            md[name].append(moment_distance((true_d,true_l),(sc.demand,sc.lead_time)))
            plan=solve_capacity_saa(sc.demand,sc.lead_time,p)
            rows=np.asarray([evaluate_plan(plan,d,l,p) for d,l in zip(true_d,true_l)],float)
            records[name].append((
                rows[:,0].mean(), cvar(rows[:,0],p.alpha),
                rows[:,1].mean(), rows[:,2].mean(), plan.reserved_capacity
            ))
    out=[]
    for name in records:
        a=np.asarray(records[name],float)
        out.append(EvaluationRow(
            name,float(np.mean(md[name])),float(a[:,0].mean()),float(a[:,1].mean()),
            float(a[:,2].mean()),float(a[:,3].mean()),float(a[:,4].mean())
        ))
    return tuple(out)
