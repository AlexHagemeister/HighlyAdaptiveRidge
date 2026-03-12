"""Convergence rate and timing experiments on synthetic DGPs.

Runs KernelHAR (and optionally HAL) across increasing sample sizes on each
DGP (Smooth, Jump, Sinusoidal) for dimensions d=1, 3, 5. Records MSE against
the true (noiseless) function and training time.

Expected runtime: ~5-15 min depending on max sample size and num_trials.
Adjust SAMPLE_SIZES and NUM_TRIALS below for quick vs full runs.

Usage:
    python experiments/run_simulations.py
"""

import sys
import os
import time
import json
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from har.kernel_har import KernelHAR
from har.data_generators import SmoothDataGenerator, JumpDataGenerator, SinusoidalDataGenerator

# -- Configuration --
SAMPLE_SIZES = [50, 100, 200, 400, 800]
DIMENSIONS = [1, 3, 5]
NUM_TRIALS = 5
TEST_SIZE = 500
SEED_BASE = 42
RUN_HAL = False  # HAL is slow for large n; set True to include

DGPS = {
    "Smooth": SmoothDataGenerator,
    "Jump": JumpDataGenerator,
    "Sinusoidal": SinusoidalDataGenerator,
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_simulations():
    results = []

    for dgp_name, dgp_cls in DGPS.items():
        for d in DIMENSIONS:
            # Fixed test set for consistent evaluation
            X_test, _ = dgp_cls.generate_data(TEST_SIZE, d, seed=SEED_BASE + 9999)
            # Noiseless truth
            f_map = {1: dgp_cls.f_1, 3: dgp_cls.f_3, 5: dgp_cls.f_5}
            f_true = f_map.get(d, lambda X: dgp_cls.f_general(X, d))
            Y_test_true = f_true(X_test)

            for n in SAMPLE_SIZES:
                for trial in range(NUM_TRIALS):
                    seed = SEED_BASE + trial * 1000 + n + d
                    X_train, Y_train = dgp_cls.generate_data(n, d, seed=seed)

                    # KernelHAR
                    model = KernelHAR(num_folds=min(5, max(2, n // 20)))
                    t0 = time.time()
                    model.fit(X_train, Y_train)
                    train_time = time.time() - t0
                    preds = model.predict(X_test)
                    mse = float(mean_squared_error(Y_test_true, preds))

                    results.append({
                        "dgp": dgp_name,
                        "d": d,
                        "n": n,
                        "trial": trial,
                        "method": "KernelHAR",
                        "mse": mse,
                        "train_time": train_time,
                        "best_lambda": model.best_lambda,
                    })

                    print(f"  {dgp_name} d={d} n={n:4d} trial={trial} "
                          f"MSE={mse:.6f} time={train_time:.2f}s "
                          f"lambda={model.best_lambda}")

                    if RUN_HAL and d <= 3:
                        from har.hal import HAL
                        hal = HAL()
                        t0 = time.time()
                        hal.fit(X_train, Y_train)
                        hal_time = time.time() - t0
                        hal_preds = hal.predict(X_test)
                        hal_mse = float(mean_squared_error(Y_test_true, hal_preds))
                        results.append({
                            "dgp": dgp_name,
                            "d": d,
                            "n": n,
                            "trial": trial,
                            "method": "HAL",
                            "mse": hal_mse,
                            "train_time": hal_time,
                            "best_lambda": None,
                        })

    return results


def main():
    print("Running convergence & timing simulations...")
    results = run_simulations()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "simulation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
