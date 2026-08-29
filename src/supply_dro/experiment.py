from .model import CostModel, empirical_demand, shifted_demand
from .optimize import evaluate, optimize_dro, optimize_nominal


def main():
    model = CostModel()
    train = empirical_demand()
    ood = shifted_demand()

    nominal = optimize_nominal(train, model)
    print("method,epsilon,order_quantity,train_mean,ood_mean,ood_cvar,ood_service_level")

    train_eval = evaluate(nominal["order_quantity"], train, model)
    ood_eval = evaluate(nominal["order_quantity"], ood, model)
    print(
        f"nominal,0,{nominal['order_quantity']:.0f},{train_eval['mean_cost']:.3f},"
        f"{ood_eval['mean_cost']:.3f},{ood_eval['cvar']:.3f},{ood_eval['service_level']:.3f}"
    )

    for epsilon in (1.0, 3.0, 6.0, 10.0):
        robust = optimize_dro(train, model, epsilon)
        train_eval = evaluate(robust["order_quantity"], train, model)
        ood_eval = evaluate(robust["order_quantity"], ood, model)
        print(
            f"dro,{epsilon:.1f},{robust['order_quantity']:.0f},{train_eval['mean_cost']:.3f},"
            f"{ood_eval['mean_cost']:.3f},{ood_eval['cvar']:.3f},{ood_eval['service_level']:.3f}"
        )


if __name__ == "__main__":
    main()
