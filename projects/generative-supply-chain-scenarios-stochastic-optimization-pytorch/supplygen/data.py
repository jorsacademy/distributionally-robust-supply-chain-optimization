from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SupplyChainDataset:
    context: np.ndarray          # [N,F]
    demand: np.ndarray           # [N,H]
    lead_time: np.ndarray        # [N,H]

    def __post_init__(self):
        x = np.asarray(self.context, dtype=float)
        d = np.asarray(self.demand, dtype=float)
        l = np.asarray(self.lead_time, dtype=float)
        if x.ndim != 2 or d.ndim != 2 or l.shape != d.shape or len(x) != len(d):
            raise ValueError("invalid dataset shapes")
        if np.any(d < 0) or np.any(l < 1):
            raise ValueError("demand must be nonnegative and lead times >=1")

    @property
    def horizon(self):
        return int(self.demand.shape[1])

def _means(x: np.ndarray, horizon: int):
    base = 42 + 20*x[:,0] + 8*x[:,1] + 6*x[:,2]*x[:,3]
    phase = 2*np.pi*x[:,4]
    promo = 1 + 0.25*np.maximum(x[:,2], 0)
    dmean, lmean = [], []
    for t in range(horizon):
        season = 8*np.sin(phase + 2*np.pi*t/horizon)
        trend = (t/(max(horizon-1,1))-.4) * (8*x[:,5])
        dm = np.maximum((base + season + trend) * promo, 2)
        lm = 1.4 + 1.0*x[:,6] + 0.35*np.cos(phase + .7*t) + 0.25*x[:,1]
        dmean.append(dm)
        lmean.append(np.maximum(lm, 1.0))
    return np.stack(dmean,1), np.stack(lmean,1)

def sample_conditional_paths(context, *, seed: int, horizon: int=6):
    rng = np.random.default_rng(seed)
    x = np.asarray(context, dtype=float)
    dm, lm = _means(x, horizon)
    n = len(x)

    common = rng.normal(size=(n,1))
    dz = rng.normal(size=(n,horizon))
    lz = rng.normal(size=(n,horizon))
    d_ar = np.zeros_like(dz)
    l_ar = np.zeros_like(lz)
    d_ar[:,0] = dz[:,0]
    l_ar[:,0] = lz[:,0]
    for t in range(1,horizon):
        d_ar[:,t] = .68*d_ar[:,t-1] + np.sqrt(1-.68**2)*dz[:,t]
        l_ar[:,t] = .55*l_ar[:,t-1] + np.sqrt(1-.55**2)*lz[:,t]

    dscale = 3.5 + 3.0*np.abs(x[:,1:2]) + 2.5*x[:,2:3]
    demand = dm + dscale*(0.45*common + 0.9*d_ar)

    disruption = rng.random((n,1)) < (0.05 + 0.08*x[:,6:7])
    surge_period = rng.integers(0,horizon,size=n)
    for i in range(n):
        if disruption[i,0]:
            t = int(surge_period[i])
            demand[i,t] += rng.uniform(12,28)
            if t+1 < horizon:
                demand[i,t+1] += rng.uniform(5,12)
            lm[i,t:] += rng.uniform(.6,1.5)

    lead_cont = lm + .4*l_ar + .18*common
    lead = np.maximum(np.rint(lead_cont), 1).astype(np.int64)
    demand = np.maximum(demand, 0)
    return demand.astype(np.float64), lead.astype(np.float64)

def generate_dataset(n_samples: int, *, seed: int, horizon: int=6):
    rng = np.random.default_rng(seed)
    x = np.column_stack([
        rng.uniform(0,1,n_samples),
        rng.normal(0,1,n_samples),
        rng.uniform(0,1,n_samples),
        rng.normal(0,1,n_samples),
        rng.uniform(0,1,n_samples),
        rng.normal(0,1,n_samples),
        rng.uniform(0,1,n_samples),
    ])
    d,l = sample_conditional_paths(x, seed=seed+9_999_991, horizon=horizon)
    return SupplyChainDataset(x,d,l)
