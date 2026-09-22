from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from .models import denormalize_paths

@dataclass(frozen=True)
class ScenarioSet:
    demand: np.ndarray
    lead_time: np.ndarray

def conditional_bootstrap(train, context, *, n_scenarios: int, seed: int=0, k: int=80):
    x=np.asarray(train.context,float)
    q=np.asarray(context,float).reshape(1,-1)
    k=min(int(k),len(x))
    nn=NearestNeighbors(n_neighbors=k).fit(x)
    idx=nn.kneighbors(q, return_distance=False)[0]
    rng=np.random.default_rng(seed)
    chosen=rng.choice(idx,size=n_scenarios,replace=True)
    return ScenarioSet(train.demand[chosen].copy(), train.lead_time[chosen].copy())

def gaussian_residual_generator(train, context, *, n_scenarios: int, seed: int=0):
    x=np.asarray(train.context,float)
    q=np.asarray(context,float).reshape(1,-1)
    nn=NearestNeighbors(n_neighbors=min(100,len(x))).fit(x)
    idx=nn.kneighbors(q,return_distance=False)[0]
    local_d=train.demand[idx]; local_l=train.lead_time[idx]
    mean_d=local_d.mean(0); mean_l=local_l.mean(0)
    resid=np.concatenate([local_d-mean_d, local_l-mean_l],1)
    cov=np.cov(resid,rowvar=False) + 1e-5*np.eye(resid.shape[1])
    rng=np.random.default_rng(seed)
    z=rng.multivariate_normal(np.zeros(resid.shape[1]),cov,size=n_scenarios)
    h=train.horizon
    d=np.maximum(mean_d+z[:,:h],0)
    l=np.maximum(np.rint(mean_l+z[:,h:]),1)
    return ScenarioSet(d,l)

@torch.no_grad()
def cvae_scenarios(result, context, *, n_scenarios: int, seed: int=0, device="cpu"):
    model=result.model.to(device)
    torch.manual_seed(seed)
    x=torch.tensor(np.repeat(np.asarray(context,float)[None,:],n_scenarios,axis=0),dtype=torch.float32,device=device)
    z=torch.randn((n_scenarios,model.latent_dim),device=device)
    raw=model.decode(x,z).cpu().numpy()
    d,l=denormalize_paths(raw,result.scaling)
    return ScenarioSet(d,l)
