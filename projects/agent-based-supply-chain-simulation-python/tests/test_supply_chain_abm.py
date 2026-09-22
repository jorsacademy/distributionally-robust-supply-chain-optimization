import math
import unittest

import numpy as np

from agent_based_supply_chain_simulation import (
    CarrierAgent,
    InventoryAgent,
    Policy,
    SUPPLIER,
    SupplyChainABM,
    candidate_policies,
    deterministic_oracle_scenario,
    estimate_policy,
    generate_scenario,
    make_scenarios,
    optimize_policy,
    paired_cost_reduction,
)


class SupplyChainABMTests(unittest.TestCase):
    def test_inventory_agent_uses_only_local_state(self):
        agent = InventoryAgent(
            name="Retailer",
            on_hand=50.0,
            backlog=10.0,
            downstream_order_queue=0.0,
            upstream_orders_outstanding=20.0,
            forecast=12.0,
        )
        self.assertTrue(math.isclose(
            agent.inventory_position(),
            60.0,
            abs_tol=1e-12,
        ))
        order = agent.desired_order(
            planning_lead_days=2.0,
            review_period_days=1,
            safety_days=3.0,
        )
        self.assertTrue(math.isclose(order, 12.0, abs_tol=1e-12))

    def test_carrier_base_lead_times_hand_oracle(self):
        scenario = deterministic_oracle_scenario()
        carrier = CarrierAgent(scenario)
        self.assertEqual(carrier.arrival_day(2, 0), 3)
        self.assertEqual(carrier.arrival_day(2, 1), 4)
        self.assertEqual(carrier.arrival_day(2, 2), 4)

    def test_constant_demand_fixture_has_near_perfect_immediate_fill(self):
        scenario = deterministic_oracle_scenario()
        policy = Policy(1, 2.0, 2.0, 2.0, 2.0, 0.4)
        result = SupplyChainABM(policy, scenario, warmup_days=0).run()
        self.assertGreaterEqual(result.customer_fill_rate, 0.99)
        self.assertLessEqual(result.end_customer_backlog, 1e-9)

    def test_scenario_generation_is_reproducible(self):
        a = generate_scenario(17, horizon_days=80, disruption_start=35, disruption_end=45)
        b = generate_scenario(17, horizon_days=80, disruption_start=35, disruption_end=45)
        np.testing.assert_array_equal(a.customer_demand, b.customer_demand)
        np.testing.assert_array_equal(a.supplier_capacity_factor, b.supplier_capacity_factor)
        np.testing.assert_array_equal(a.carrier_delay_extra_days, b.carrier_delay_extra_days)

    def test_supplier_confirmed_orders_never_exceed_external_capacity(self):
        scenario = generate_scenario(19, horizon_days=80, disruption_start=35, disruption_end=45)
        policy = Policy(1, 3.0, 3.0, 1.0, 1.0, 0.2)
        sim = SupplyChainABM(policy, scenario, warmup_days=10)
        sim.run()
        capacity = sim.NOMINAL_EXTERNAL_SUPPLY_CAPACITY * scenario.supplier_capacity_factor
        self.assertTrue(np.all(sim.daily_stage_orders[:, SUPPLIER] <= capacity + 1e-9))

    def test_candidate_policy_grid_is_complete_and_unique(self):
        policies = candidate_policies()
        self.assertEqual(len(policies), 54)
        self.assertEqual(len(set(policies)), 54)

    def test_policy_estimation_is_reproducible_under_crn(self):
        scenarios = make_scenarios(range(200, 205), horizon_days=80)
        policy = Policy(1, 3.0, 3.0, 1.0, 1.0, 0.2)
        a = estimate_policy(policy, scenarios, warmup_days=10)
        b = estimate_policy(policy, scenarios, warmup_days=10)
        self.assertEqual(a, b)

    def test_paired_cost_interval_contains_paired_mean(self):
        scenarios = make_scenarios(range(500, 508))
        selected = Policy(1, 3.0, 3.0, 1.0, 1.0, 0.2)
        baseline = Policy(1, 1.0, 1.0, 1.0, 1.0, 0.65)
        mean, low, high = paired_cost_reduction(selected, baseline, scenarios)
        self.assertLessEqual(low, mean)
        self.assertLessEqual(mean, high)

    def test_short_policy_optimization_returns_declared_candidate(self):
        result = optimize_policy(selection_replications=4, validation_replications=6, seed=9)
        self.assertEqual(result.candidate_count, 54)
        self.assertIn(result.selected.policy, candidate_policies())
        self.assertGreaterEqual(result.selected_validation.mean_fill_rate, 0.0)
        self.assertLessEqual(result.selected_validation.mean_fill_rate, 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
