import unittest
import numpy as np
from supplygen import (
    generate_dataset, train_cvae,
    conditional_bootstrap, gaussian_residual_generator, cvae_scenarios,
    PlanningParameters, solve_capacity_saa, evaluate_plan, cvar
)

class SupplyGenTests(unittest.TestCase):
    def test_data_reproducible(self):
        a=generate_dataset(20,seed=11,horizon=4)
        b=generate_dataset(20,seed=11,horizon=4)
        np.testing.assert_allclose(a.demand,b.demand)
        np.testing.assert_allclose(a.lead_time,b.lead_time)

    def test_paths_valid(self):
        ds=generate_dataset(30,seed=12,horizon=5)
        self.assertTrue(np.all(ds.demand>=0))
        self.assertTrue(np.all(ds.lead_time>=1))

    def test_bootstrap_and_gaussian_shapes(self):
        ds=generate_dataset(80,seed=13,horizon=4)
        for sc in [
            conditional_bootstrap(ds,ds.context[0],n_scenarios=17,seed=1),
            gaussian_residual_generator(ds,ds.context[0],n_scenarios=17,seed=1),
        ]:
            self.assertEqual(sc.demand.shape,(17,4))
            self.assertEqual(sc.lead_time.shape,(17,4))
            self.assertTrue(np.all(sc.lead_time>=1))

    def test_cvae_trains_and_samples(self):
        tr=generate_dataset(60,seed=14,horizon=4)
        va=generate_dataset(20,seed=15,horizon=4)
        r=train_cvae(tr,va,seed=2,epochs=2,batch_size=30,latent_dim=4,hidden=32)
        sc=cvae_scenarios(r,va.context[0],n_scenarios=11,seed=7)
        self.assertEqual(sc.demand.shape,(11,4))
        self.assertTrue(np.all(sc.demand>=0))

    def test_capacity_solver_matches_bruteforce_declared_objective(self):
        p=PlanningParameters(max_reserved_capacity=8,alpha=.8,cvar_weight=.2)
        D=np.array([[20,25,22],[35,30,28],[18,40,25]],float)
        L=np.array([[1,1,2],[2,1,2],[1,2,1]],float)
        sol=solve_capacity_saa(D,L,p)
        vals=[]
        for r in range(9):
            costs=np.array([evaluate_plan(type(sol)(r,0,"",3),d,l,p)[0] for d,l in zip(D,L)])
            obj=costs.mean()+p.cvar_weight*cvar(costs,p.alpha)
            vals.append(obj)
        self.assertAlmostEqual(sol.objective,min(vals),places=8)
        self.assertEqual(sol.reserved_capacity,float(np.argmin(vals)))

    def test_more_reserve_returns_finite_operational_evaluation(self):
        p=PlanningParameters(max_reserved_capacity=5)
        d=np.array([80.,60.,70.]); l=np.array([1.,1.,1.])
        low=type("P",(),{"reserved_capacity":0.0})()
        high=type("P",(),{"reserved_capacity":5.0})()
        self.assertTrue(np.isfinite(evaluate_plan(low,d,l,p)[0]))
        self.assertTrue(np.isfinite(evaluate_plan(high,d,l,p)[0]))

if __name__=="__main__":
    unittest.main()
