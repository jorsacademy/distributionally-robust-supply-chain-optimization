from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch
from .models import ConditionalVAE, fit_scaling, normalize_paths, vae_loss

@dataclass(frozen=True)
class TrainResult:
    model: ConditionalVAE
    scaling: object
    history: tuple

def train_cvae(train, validation, *, seed=42, epochs=35, batch_size=128, latent_dim=8, hidden=96, lr=1e-3, device="cpu"):
    np.random.seed(seed); torch.manual_seed(seed)
    scaling=fit_scaling(train.demand, train.lead_time)
    xtr=torch.tensor(train.context,dtype=torch.float32)
    ytr=torch.tensor(normalize_paths(train.demand,train.lead_time,scaling),dtype=torch.float32)
    xv=torch.tensor(validation.context,dtype=torch.float32,device=device)
    yv=torch.tensor(normalize_paths(validation.demand,validation.lead_time,scaling),dtype=torch.float32,device=device)
    model=ConditionalVAE(train.context.shape[1], train.horizon, latent_dim, hidden).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    rng=np.random.default_rng(seed)
    best=float("inf"); best_state=None; history=[]
    for ep in range(1,epochs+1):
        model.train()
        order=rng.permutation(len(xtr)); losses=[]
        for st in range(0,len(order),batch_size):
            idx=order[st:st+batch_size]
            xb=xtr[idx].to(device); yb=ytr[idx].to(device)
            recon,mu,logvar=model(xb,yb)
            loss,rec,kl=vae_loss(recon,yb,mu,logvar)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            mu,logvar=model.encode(xv,yv)
            recon=model.decode(xv,mu)
            val=float(torch.mean((recon-yv)**2).item())
        if val<best:
            best=val
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append((ep,float(np.mean(losses)),val))
    model.load_state_dict(best_state); model.eval()
    return TrainResult(model,scaling,tuple(history))
