from __future__ import annotations
import numpy as np

def path_moments(demand, lead):
    d=np.asarray(demand,float); l=np.asarray(lead,float)
    return {
        "demand_mean":float(d.mean()),
        "demand_std":float(d.std()),
        "lead_mean":float(l.mean()),
        "lead_std":float(l.std()),
        "demand_lag1":_lag1(d),
        "lead_lag1":_lag1(l),
        "demand_lead_corr":_flat_corr(d,l),
        "demand_p95":float(np.quantile(d,.95)),
        "lead_p95":float(np.quantile(l,.95)),
    }

def _lag1(a):
    if a.shape[1]<2: return 0.0
    x=a[:,:-1].reshape(-1); y=a[:,1:].reshape(-1)
    return _flat_corr(x,y)

def _flat_corr(a,b):
    x=np.asarray(a,float).reshape(-1); y=np.asarray(b,float).reshape(-1)
    if x.std()<1e-12 or y.std()<1e-12: return 0.0
    return float(np.corrcoef(x,y)[0,1])

def moment_distance(reference, generated):
    r=path_moments(*reference); g=path_moments(*generated)
    diffs=[]
    for k in r.keys():
        scale=max(abs(r[k]),1.0)
        diffs.append(abs(g[k]-r[k])/scale)
    return float(np.mean(diffs))

def cvar(values, alpha=.9):
    v=np.asarray(values,float)
    q=np.quantile(v,alpha,method="higher")
    tail=v[v>=q-1e-12]
    return float(tail.mean())
