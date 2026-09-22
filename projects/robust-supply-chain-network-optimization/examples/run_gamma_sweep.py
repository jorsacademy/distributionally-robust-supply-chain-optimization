import json

from robust_scn import demo_instance, gamma_sweep


if __name__ == "__main__":
    results = gamma_sweep(
        demo_instance(),
        gammas=[0, 1, 2, 3, 4, 6],
        n_scenarios=3000,
        seed=2026,
    )
    print(json.dumps(results, indent=2))
