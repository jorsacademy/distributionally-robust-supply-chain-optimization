from .network import default_network, demand_scenarios, disruption_scenarios
from .network_optimize import (
    evaluate_disruption_policy,
    evaluate_network_policy,
    optimize_classical_robust_network,
    optimize_dro_network,
    optimize_nominal_network,
)


def _capacity_text(capacity):
    return "/".join(f"{value:.0f}" for value in capacity)


def _report(method, epsilon, solution, train, ood, stress, data):
    train_eval = evaluate_network_policy(solution["capacity"], train, data)
    ood_eval = evaluate_network_policy(solution["capacity"], ood, data)
    stress_eval = evaluate_disruption_policy(solution["capacity"], stress, data)
    print(
        f"{method},{epsilon},{_capacity_text(solution['capacity'])},"
        f"{train_eval['mean_total_cost']:.3f},{ood_eval['mean_total_cost']:.3f},"
        f"{ood_eval['p90_total_cost']:.3f},{ood_eval['worst_total_cost']:.3f},"
        f"{stress_eval['mean_stress_cost']:.3f},{stress_eval['worst_stress_cost']:.3f},"
        f"{stress_eval['worst_stress_scenario']}"
    )


def main():
    data = default_network()
    train = demand_scenarios(seed=7, n=6)
    ood = demand_scenarios(seed=77, n=40, shift=12.0)
    stress = disruption_scenarios()

    print(
        "method,epsilon,capacity,train_mean,ood_mean,ood_p90,ood_worst,"
        "stress_mean,stress_worst,worst_stress_scenario"
    )

    nominal = optimize_nominal_network(train, data, step=20.0)
    _report("nominal", 0, nominal, train, ood, stress, data)

    for epsilon in (4.0, 8.0, 12.0):
        robust = optimize_dro_network(train, data, epsilon=epsilon, step=20.0)
        _report("wasserstein_dro", f"{epsilon:.1f}", robust, train, ood, stress, data)

    classical = optimize_classical_robust_network(stress, data, step=20.0)
    _report("classical_robust", "stress_set", classical, train, ood, stress, data)


if __name__ == "__main__":
    main()
