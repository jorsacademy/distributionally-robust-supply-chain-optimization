from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

class ConditionalVAE(nn.Module):
    def __init__(self, context_dim: int, horizon: int, latent_dim: int=8, hidden: int=96):
        super().__init__()
        self.context_dim=context_dim
        self.horizon=horizon
        self.latent_dim=latent_dim
        ydim=2*horizon
        self.encoder=nn.Sequential(
            nn.Linear(context_dim+ydim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu=nn.Linear(hidden, latent_dim)
        self.logvar=nn.Linear(hidden, latent_dim)
        self.decoder=nn.Sequential(
            nn.Linear(context_dim+latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ydim)
        )

    def encode(self,x,y):
        h=self.encoder(torch.cat([x,y],-1))
        return self.mu(h), self.logvar(h)

    def decode(self,x,z):
        return self.decoder(torch.cat([x,z],-1))

    def forward(self,x,y):
        mu,logvar=self.encode(x,y)
        std=torch.exp(.5*logvar)
        z=mu+std*torch.randn_like(std)
        return self.decode(x,z),mu,logvar

@dataclass(frozen=True)
class Scaling:
    demand_mean: np.ndarray
    demand_std: np.ndarray
    lead_mean: np.ndarray
    lead_std: np.ndarray

def fit_scaling(demand, lead):
    d=np.asarray(demand,float); l=np.asarray(lead,float)
    return Scaling(
        d.mean(0), np.maximum(d.std(0),1.0),
        l.mean(0), np.maximum(l.std(0),.25)
    )

def normalize_paths(demand, lead, scaling: Scaling):
    d=(np.asarray(demand)-scaling.demand_mean)/scaling.demand_std
    l=(np.asarray(lead)-scaling.lead_mean)/scaling.lead_std
    return np.concatenate([d,l],1)

def denormalize_paths(raw, scaling: Scaling):
    raw=np.asarray(raw,float)
    h=len(scaling.demand_mean)
    d=raw[:,:h]*scaling.demand_std+scaling.demand_mean
    l=raw[:,h:]*scaling.lead_std+scaling.lead_mean
    return np.maximum(d,0), np.maximum(np.rint(l),1)

def vae_loss(recon,target,mu,logvar,beta=.02):
    rec=torch.mean((recon-target)**2)
    kl=-.5*torch.mean(1+logvar-mu.square()-logvar.exp())
    return rec+beta*kl, rec, kl
