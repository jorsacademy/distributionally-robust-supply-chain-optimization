from __future__ import annotations
import argparse
import numpy as np
import torch

from supplygen import generate_dataset, train_cvae, evaluate_generators

def self_test():
    train=generate_dataset(40,seed=1,horizon=4)
    val=generate_dataset(12,seed=2,horizon=4)
    result=train_cvae(train,val,seed=3,epochs=2,batch_size=20,latent_dim=4,hidden=32)
    assert len(result.history)==2
    print("Generative supply-chain self-test: OK")

def main(args):
    train=generate_dataset(args.train_samples,seed=args.seed,horizon=args.horizon)
    val=generate_dataset(args.validation_samples,seed=args.seed+1_000_000,horizon=args.horizon)
    test=generate_dataset(args.test_contexts,seed=args.seed+2_000_000,horizon=args.horizon)
    result=train_cvae(
        train,val,seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,
        latent_dim=args.latent_dim,hidden=args.hidden,device=args.device
    )
    best=min(x[2] for x in result.history)
    print(f"best validation reconstruction MSE: {best:.5f}")
    rows=evaluate_generators(
        train,result,test.context,
        n_scenarios=args.scenarios,eval_scenarios=args.eval_scenarios,
        seed=args.seed+3_000_000,device=args.device
    )
    print(f"{'method':<24}{'moment dist':>13}{'mean cost':>13}{'CVaR':>13}{'fill':>10}{'stockout':>11}{'reserve':>10}")
    for r in rows:
        print(f"{r.method:<24}{r.moment_distance:13.4f}{r.mean_cost:13.2f}{r.cvar_cost:13.2f}{r.mean_fill_rate:10.4f}{r.mean_stockout_rate:11.4f}{r.mean_reserved_capacity:10.2f}")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--horizon",type=int,default=6)
    p.add_argument("--train-samples",type=int,default=600)
    p.add_argument("--validation-samples",type=int,default=160)
    p.add_argument("--test-contexts",type=int,default=8)
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--batch-size",type=int,default=96)
    p.add_argument("--latent-dim",type=int,default=8)
    p.add_argument("--hidden",type=int,default=96)
    p.add_argument("--scenarios",type=int,default=96)
    p.add_argument("--eval-scenarios",type=int,default=192)
    p.add_argument("--device",default="cpu")
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    self_test() if args.self_test else main(args)
