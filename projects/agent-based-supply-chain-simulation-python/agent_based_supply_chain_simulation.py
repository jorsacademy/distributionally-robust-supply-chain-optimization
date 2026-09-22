from __future__ import annotations
import argparse, math
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy.stats import t as student_t

RETAILER, WAREHOUSE, FACTORY, SUPPLIER = range(4)
STAGE_NAMES = ('Retailer','Warehouse','Factory','Supplier')

@dataclass(frozen=True)
class Policy:
    review_period_days:int
    safety_days_retailer:float
    safety_days_warehouse:float
    safety_days_factory:float
    safety_days_supplier:float
    smoothing_alpha:float
    def __post_init__(self):
        if self.review_period_days<1: raise ValueError('review period must be >= 1')
        if min(self.safety_days)<0: raise ValueError('safety days must be nonnegative')
        if not 0<self.smoothing_alpha<=1: raise ValueError('smoothing_alpha must be in (0,1]')
    @property
    def safety_days(self):
        return (self.safety_days_retailer,self.safety_days_warehouse,self.safety_days_factory,self.safety_days_supplier)

@dataclass(frozen=True)
class SupplyChainScenario:
    customer_demand:np.ndarray
    supplier_capacity_factor:np.ndarray
    carrier_delay_extra_days:np.ndarray
    horizon_days:int
    disruption_start:int
    disruption_end:int
    def __post_init__(self):
        H=self.horizon_days
        if self.customer_demand.shape!=(H,) or self.supplier_capacity_factor.shape!=(H,) or self.carrier_delay_extra_days.shape!=(H,3): raise ValueError('scenario shape mismatch')
        if np.any(self.customer_demand<0) or np.any(self.carrier_delay_extra_days<0): raise ValueError('negative scenario value')
        if np.any((self.supplier_capacity_factor<0)|(self.supplier_capacity_factor>1)): raise ValueError('invalid supplier capacity factor')
        if not 0<=self.disruption_start<self.disruption_end<=H: raise ValueError('invalid disruption window')

@dataclass
class InventoryAgent:
    name:str; on_hand:float; backlog:float; downstream_order_queue:float; upstream_orders_outstanding:float; forecast:float
    def inventory_position(self): return self.on_hand+self.upstream_orders_outstanding-self.backlog
    def update_forecast(self,observation,alpha): self.forecast=alpha*float(observation)+(1-alpha)*self.forecast
    def desired_order(self,*,planning_lead_days,review_period_days,safety_days):
        target=self.forecast*(planning_lead_days+review_period_days+safety_days)
        return max(0.0,target-self.inventory_position())

@dataclass(frozen=True)
class ReplicationResult:
    total_cost:float; customer_fill_rate:float; mean_customer_backlog:float; mean_total_inventory:float; bullwhip_ratio:float; recovery_days:float; end_customer_backlog:float; stage_order_variance:tuple; customer_demand_variance:float
@dataclass(frozen=True)
class PolicyEstimate:
    policy:Policy; mean_cost:float; std_cost:float; mean_fill_rate:float; mean_backlog:float; mean_inventory:float; mean_bullwhip:float; mean_recovery_days:float; replications:int
@dataclass(frozen=True)
class OptimizationResult:
    selected:PolicyEstimate; baseline_validation:PolicyEstimate; selected_validation:PolicyEstimate; candidate_count:int; paired_cost_reduction_mean:float; paired_cost_reduction_ci95_low:float; paired_cost_reduction_ci95_high:float

class CarrierAgent:
    BASE_LEAD_TIMES=(1,2,2)
    def __init__(self,scenario): self.scenario=scenario
    def arrival_day(self,dispatch_day,link):
        if link not in (0,1,2): raise ValueError('invalid link')
        return dispatch_day+self.BASE_LEAD_TIMES[link]+int(self.scenario.carrier_delay_extra_days[dispatch_day,link])

class SupplyChainABM:
    HOLDING_COST=np.array((0.32,0.22,0.18,0.12))
    BACKLOG_COST=np.array((4.5,2.7,2.0,1.6))
    ORDER_COST=np.array((0.04,0.035,0.03,0.025))
    INITIAL_INVENTORY=(140.,220.,280.,330.)
    NOMINAL_EXTERNAL_SUPPLY_CAPACITY=125.
    PLANNING_LEAD_DAYS=(2.,3.,3.,2.)
    def __init__(self,policy,scenario,*,warmup_days=20):
        if not 0<=warmup_days<scenario.horizon_days: raise ValueError('invalid warmup')
        self.policy,self.scenario,self.warmup_days=policy,scenario,warmup_days
        self.carrier=CarrierAgent(scenario)
        f=float(max(np.mean(scenario.customer_demand[:min(10,scenario.horizon_days)]),1.0))
        self.stages=[InventoryAgent(STAGE_NAMES[i],self.INITIAL_INVENTORY[i],0.,0.,0.,f) for i in range(4)]
        self.arrivals=defaultdict(list); self.external_supplier_arrivals=defaultdict(float)
        H=scenario.horizon_days
        self.daily_customer_demand=np.zeros(H); self.daily_customer_immediate_served=np.zeros(H); self.daily_customer_backlog=np.zeros(H)
        self.daily_total_inventory=np.zeros(H); self.daily_stage_inventory=np.zeros((H,4)); self.daily_stage_backlog=np.zeros((H,4)); self.daily_stage_orders=np.zeros((H,4)); self.daily_stage_shipments=np.zeros((H,4))
    def _receive_arrivals(self,d):
        for dst,q in self.arrivals.pop(d,[]):
            a=self.stages[dst]; a.on_hand+=q; a.upstream_orders_outstanding=max(0.,a.upstream_orders_outstanding-q)
        q=self.external_supplier_arrivals.pop(d,0.)
        if q:
            a=self.stages[SUPPLIER]; a.on_hand+=q; a.upstream_orders_outstanding=max(0.,a.upstream_orders_outstanding-q)
    def _ship(self,u,d):
        a=self.stages[u]; req=a.downstream_order_queue+a.backlog; q=min(a.on_hand,req); a.on_hand-=q; a.backlog=req-q; a.downstream_order_queue=0.
        if q:
            arr=self.carrier.arrival_day(d,3-u)
            if arr<self.scenario.horizon_days: self.arrivals[arr].append((u-1,q))
        self.daily_stage_shipments[d,u]=q
    def _serve_customer(self,d):
        a=self.stages[RETAILER]; dem=float(self.scenario.customer_demand[d]); old=a.backlog; self.daily_customer_demand[d]=dem
        old_served=min(a.on_hand,old); a.on_hand-=old_served; old-=old_served
        now=min(a.on_hand,dem); a.on_hand-=now; a.backlog=old+(dem-now); self.daily_customer_immediate_served[d]=now; self.daily_stage_shipments[d,RETAILER]=old_served+now
    def _place_orders(self,d):
        if d%self.policy.review_period_days: return
        for i,a in enumerate(self.stages):
            q=a.desired_order(planning_lead_days=self.PLANNING_LEAD_DAYS[i],review_period_days=self.policy.review_period_days,safety_days=self.policy.safety_days[i])
            if i<SUPPLIER:
                self.daily_stage_orders[d,i]=q; a.upstream_orders_outstanding+=q; self.stages[i+1].downstream_order_queue+=q
            else:
                q=min(q,self.NOMINAL_EXTERNAL_SUPPLY_CAPACITY*self.scenario.supplier_capacity_factor[d]); self.daily_stage_orders[d,i]=q; a.upstream_orders_outstanding+=q
                if q and d+1<self.scenario.horizon_days: self.external_supplier_arrivals[d+1]+=q
    def _update_forecasts(self,d):
        alpha=self.policy.smoothing_alpha; self.stages[RETAILER].update_forecast(self.daily_customer_demand[d],alpha)
        for u in (WAREHOUSE,FACTORY,SUPPLIER): self.stages[u].update_forecast(self.daily_stage_orders[d,u-1],alpha)
    def _record(self,d):
        for i,a in enumerate(self.stages): self.daily_stage_inventory[d,i]=a.on_hand; self.daily_stage_backlog[d,i]=a.backlog
        self.daily_customer_backlog[d]=self.stages[RETAILER].backlog; self.daily_total_inventory[d]=self.daily_stage_inventory[d].sum()
    def _recovery(self):
        start=self.scenario.disruption_end; threshold=.01*max(float(np.mean(self.scenario.customer_demand)),1.); streak=0
        for d in range(start,self.scenario.horizon_days):
            streak=streak+1 if self.daily_customer_backlog[d]<=threshold else 0
            if streak>=5: return float(d-4-start)
        return float(self.scenario.horizon_days-start+1)
    def run(self):
        H=self.scenario.horizon_days
        for d in range(H):
            self._receive_arrivals(d)
            for u in (SUPPLIER,FACTORY,WAREHOUSE): self._ship(u,d)
            self._serve_customer(d); self._place_orders(d); self._update_forecasts(d); self._record(d)
        m=slice(self.warmup_days,H); days=H-self.warmup_days
        dem=self.daily_customer_demand[m]; immediate=self.daily_customer_immediate_served[m]; inv=self.daily_stage_inventory[m]; bg=self.daily_stage_backlog[m]; orders=self.daily_stage_orders[m]
        fill=float(immediate.sum()/max(dem.sum(),1e-12)); hold=float((inv*self.HOLDING_COST).sum()); back=float((bg*self.BACKLOG_COST).sum()); oc=float((orders*self.ORDER_COST).sum()); terminal=float((self.daily_stage_backlog[-1]*self.BACKLOG_COST).sum())
        total=(hold+back+oc+terminal)/days; dv=float(np.var(dem,ddof=1)) if len(dem)>1 else 0.; ovs=tuple(float(np.var(orders[:,i],ddof=1)) if len(orders)>1 else 0. for i in range(4)); bull=max(ovs)/dv if dv>1e-12 else 0.
        return ReplicationResult(total,fill,float(np.mean(self.daily_customer_backlog[m])),float(np.mean(self.daily_total_inventory[m])),float(bull),self._recovery(),float(self.daily_customer_backlog[-1]),ovs,dv)

def generate_scenario(seed,*,horizon_days=120,disruption_start=55,disruption_end=72):
    rng=np.random.default_rng(seed); day=np.arange(horizon_days); weekly=1+.08*np.sin(2*np.pi*day/7); eps=rng.normal(0,.08,horizon_days); demand=42*weekly*np.exp(eps-.5*.08**2)
    surge=np.ones(horizon_days); surge[disruption_start:disruption_end]*=1.32; end=min(horizon_days,disruption_end+10)
    if end>disruption_end: surge[disruption_end:end]*=np.linspace(1.2,1.,end-disruption_end,endpoint=False)
    demand*=surge; cap=np.ones(horizon_days); cap[disruption_start:disruption_start+4]=.15; cap[disruption_start+4:disruption_start+9]=.45; cap[disruption_start+9:disruption_end]=.75; cap[rng.random(horizon_days)<.025]*=.75
    delays=np.zeros((horizon_days,3),dtype=int)
    for d in range(horizon_days):
        for link in range(3):
            if rng.random()<.04: delays[d,link]+=1
            if disruption_start<=d<disruption_end and rng.random()<.25+.08*link: delays[d,link]+=int(rng.integers(1,4))
    return SupplyChainScenario(demand.astype(float),cap.astype(float),delays,horizon_days,disruption_start,disruption_end)

def make_scenarios(seeds,*,horizon_days=120): return [generate_scenario(int(s),horizon_days=horizon_days) for s in seeds]
def estimate_policy(policy,scenarios,*,warmup_days=20):
    if len(scenarios)<2: raise ValueError('at least two scenarios required')
    r=[SupplyChainABM(policy,s,warmup_days=warmup_days).run() for s in scenarios]; arr=lambda f:np.array([getattr(x,f) for x in r],float)
    c,fi,b,i,bu,re=map(arr,('total_cost','customer_fill_rate','mean_customer_backlog','mean_total_inventory','bullwhip_ratio','recovery_days'))
    return PolicyEstimate(policy,float(c.mean()),float(c.std(ddof=1)),float(fi.mean()),float(b.mean()),float(i.mean()),float(bu.mean()),float(re.mean()),len(r))
def paired_cost_reduction(selected_policy,baseline_policy,scenarios,*,warmup_days=20):
    if len(scenarios)<2: raise ValueError('at least two paired scenarios required')
    d=np.array([SupplyChainABM(baseline_policy,s,warmup_days=warmup_days).run().total_cost-SupplyChainABM(selected_policy,s,warmup_days=warmup_days).run().total_cost for s in scenarios]); mean=float(d.mean()); half=float(student_t.ppf(.975,len(d)-1)*d.std(ddof=1)/math.sqrt(len(d))); return mean,mean-half,mean+half
def candidate_policies(): return tuple(Policy(r,ds,ds,us,us,a) for r in (1,2) for ds in (1.,2.,3.) for us in (1.,2.,3.) for a in (.2,.4,.65))
def score_estimate(e): return e.mean_cost+5000*max(0,.985-e.mean_fill_rate)+5*math.log1p(e.mean_bullwhip)+2*e.mean_recovery_days
def optimize_policy(*,selection_replications=16,validation_replications=36,seed=42,baseline=Policy(1,1.,1.,1.,1.,.65)):
    if min(selection_replications,validation_replications)<2: raise ValueError('need at least two replications')
    sel=make_scenarios(seed+i for i in range(selection_replications)); estimates=[estimate_policy(p,sel) for p in candidate_policies()]; estimates.sort(key=lambda e:(score_estimate(e),e.mean_cost,-e.mean_fill_rate,e.mean_bullwhip)); chosen=estimates[0]
    val=make_scenarios(seed+100000+i for i in range(validation_replications)); sv=estimate_policy(chosen.policy,val); bv=estimate_policy(baseline,val); mean,lo,hi=paired_cost_reduction(chosen.policy,baseline,val)
    return OptimizationResult(chosen,bv,sv,len(candidate_policies()),mean,lo,hi)
def deterministic_oracle_scenario():
    H=12; return SupplyChainScenario(np.full(H,10.),np.ones(H),np.zeros((H,3),int),H,6,7)
def self_test():
    a=generate_scenario(7,horizon_days=80,disruption_start=35,disruption_end=45); b=generate_scenario(7,horizon_days=80,disruption_start=35,disruption_end=45); assert np.array_equal(a.customer_demand,b.customer_demand) and np.array_equal(a.supplier_capacity_factor,b.supplier_capacity_factor) and np.array_equal(a.carrier_delay_extra_days,b.carrier_delay_extra_days)
    o=deterministic_oracle_scenario(); c=CarrierAgent(o); assert (c.arrival_day(2,0),c.arrival_day(2,1),c.arrival_day(2,2))==(3,4,4)
    p=Policy(1,2.,2.,2.,2.,.4); r=SupplyChainABM(p,o,warmup_days=0).run(); assert r.customer_fill_rate>=.99 and r.end_customer_backlog<=1e-9
    ps=candidate_policies(); assert len(ps)==54 and len(set(ps))==54
    ss=make_scenarios(range(200,204),horizon_days=80); assert estimate_policy(p,ss,warmup_days=10)==estimate_policy(p,ss,warmup_days=10)
    x=optimize_policy(selection_replications=4,validation_replications=6,seed=9); assert x.selected.policy in ps and x.paired_cost_reduction_ci95_low<=x.paired_cost_reduction_mean<=x.paired_cost_reduction_ci95_high
    print('Agent-based supply-chain simulation self-test: OK')
def print_result(x):
    s,b=x.selected_validation,x.baseline_validation; print('='*90); print('AGENT-BASED SUPPLY-CHAIN RESILIENCE SIMULATION + POLICY OPTIMIZATION'); print('='*90); print(f'Candidate policies checked       : {x.candidate_count}\n'); print('Selected policy'); print(f'  review period                  : {s.policy.review_period_days} day(s)\n  retailer/warehouse safety      : {s.policy.safety_days_retailer:.1f} demand-days\n  factory/supplier safety        : {s.policy.safety_days_factory:.1f} demand-days\n  forecast smoothing alpha       : {s.policy.smoothing_alpha:.2f}\n  validation mean cost           : {s.mean_cost:.3f}\n  validation fill rate           : {100*s.mean_fill_rate:.3f}%\n  mean customer backlog          : {s.mean_backlog:.3f}\n  mean total inventory           : {s.mean_inventory:.3f}\n  mean bullwhip ratio            : {s.mean_bullwhip:.3f}\n  mean recovery time             : {s.mean_recovery_days:.3f} days\n'); print('Baseline policy'); print(f'  validation mean cost           : {b.mean_cost:.3f}\n  validation fill rate           : {100*b.mean_fill_rate:.3f}%\n  mean customer backlog          : {b.mean_backlog:.3f}\n  mean total inventory           : {b.mean_inventory:.3f}\n  mean bullwhip ratio            : {b.mean_bullwhip:.3f}\n  mean recovery time             : {b.mean_recovery_days:.3f} days\n'); print(f'Paired cost reduction            : {x.paired_cost_reduction_mean:.3f} [95% CI {x.paired_cost_reduction_ci95_low:.3f}, {x.paired_cost_reduction_ci95_high:.3f}]'); print('\nThe selected policy is the exact minimizer of the declared composite sample score over the finite 54-policy grid. It is not a global optimum over all possible decentralized supply-chain policies.')
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--selection-replications',type=int,default=16); p.add_argument('--validation-replications',type=int,default=36); p.add_argument('--seed',type=int,default=42); return p.parse_args()
if __name__=='__main__':
    a=parse_args(); self_test() if a.self_test else print_result(optimize_policy(selection_replications=a.selection_replications,validation_replications=a.validation_replications,seed=a.seed))
