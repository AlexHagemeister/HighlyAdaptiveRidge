"""UCI regression benchmark experiments.

Evaluates KernelHAR on UCI datasets in data/. For each dataset, runs
repeated train/test splits and records MSE and training time.

For large datasets (n > MAX_N), a random subsample is used.

Usage:
    python experiments/run_benchmarks.py
"""

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from har.kernel_har import KernelHAR

# -- Configuration --
NUM_SPLITS = 5
TEST_FRACTION = 0.2
MAX_N = 1000  # subsample large datasets
SEED = 42

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Datasets to run (all CSVs in data/, last column = target)
DATASETS = [
    "yacht", "boston", "energy", "concrete", "wine",
    "power", "kin8nm", "naval", "protein",
]


def load_dataset(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(np.float64)
    Y = df.iloc[:, -1].values.astype(np.float64)
    return X, Y


def run_benchmarks():
    results = []

    for name in DATASETS:
        try:
            X, Y = load_dataset(name)
        except FileNotFoundError:
            print(f"  {name}: file not found, skipping")
            continue

        n_total, p = X.shape
        print(f"\n{name}: n={n_total}, p={p}")

        for split in range(NUM_SPLITS):
            seed = SEED + split

            # Subsample if too large
            if n_total > MAX_N:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_total, MAX_N, replace=False)
                X_sub, Y_sub = X[idx], Y[idx]
            else:
                X_sub, Y_sub = X, Y

            X_train, X_test, Y_train, Y_test = train_test_split(
                X_sub, Y_sub, test_size=TEST_FRACTION, random_state=seed
            )

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = KernelHAR()
            t0 = time.time()
            model.fit(X_train, Y_train)
            train_time = time.time() - t0

            preds = model.predict(X_test)
            mse = float(mean_squared_error(Y_test, preds))

            results.append({
                "dataset": name,
                "n_used": X_train.shape[0],
                "p": p,
                "split": split,
                "mse": mse,
                "train_time": train_time,
                "best_lambda": model.best_lambda,
            })

            print(f"  split={split} n_train={X_train.shape[0]} "
                  f"MSE={mse:.4f} time={train_time:.2f}s")

    return results


def main():
    print("Running UCI benchmark experiments...")
    results = run_benchmarks()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
