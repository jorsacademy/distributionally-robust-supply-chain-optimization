from .network import default_network, demand_scenarios
from .network_optimize import evaluate_network_policy, optimize_dro_network, optimize_nominal_network


def _capacity_text(capacity):
    return "/".join(f"{value:.0f}" for value in capacity)


def main():
    data = default_network()
    train = demand_scenarios(seed=7, n=6)
    ood = demand_scenarios(seed=77, n=40, shift=12.0)

    nominal = optimize_nominal_network(train, data, step=20.0)
    nominal_train = evaluate_network_policy(nominal["capacity"], train, data)
    nominal_ood = evaluate_network_policy(nominal["capacity"], ood, data)

    print("method,epsilon,capacity,train_mean,ood_mean,ood_p90,ood_worst")
    print(
        f"nominal,0,{_capacity_text(nominal['capacity'])},"
        f"{nominal_train['mean_total_cost']:.3f},{nominal_ood['mean_total_cost']:.3f},"
        f"{nominal_ood['p90_total_cost']:.3f},{nominal_ood['worst_total_cost']:.3f}"
    )

    for epsilon in (4.0, 8.0, 12.0):
        robust = optimize_dro_network(train, data, epsilon=epsilon, step=20.0)
        train_eval = evaluate_network_policy(robust["capacity"], train, data)
        ood_eval = evaluate_network_policy(robust["capacity"], ood, data)
        print(
            f"dro,{epsilon:.1f},{_capacity_text(robust['capacity'])},"
            f"{train_eval['mean_total_cost']:.3f},{ood_eval['mean_total_cost']:.3f},"
            f"{ood_eval['p90_total_cost']:.3f},{ood_eval['worst_total_cost']:.3f}"
        )


if __name__ == "__main__":
    main()
