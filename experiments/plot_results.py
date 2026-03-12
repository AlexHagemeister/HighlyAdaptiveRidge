"""Generate figures from simulation and benchmark results.

Reads JSON results from results/ and produces plots in results/figures/.

Usage:
    python experiments/plot_results.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def plot_convergence(results):
    """Plot MSE vs n for each DGP and dimension, with n^{-1/3} reference line."""
    import matplotlib.pyplot as plt

    dgps = sorted(set(r["dgp"] for r in results))
    dims = sorted(set(r["d"] for r in results))

    fig, axes = plt.subplots(len(dgps), len(dims), figsize=(4 * len(dims), 3.5 * len(dgps)),
                             squeeze=False, sharex=True)

    for i, dgp in enumerate(dgps):
        for j, d in enumerate(dims):
            ax = axes[i][j]
            subset = [r for r in results if r["dgp"] == dgp and r["d"] == d
                      and r["method"] == "KernelHAR"]

            # Aggregate by n
            ns = sorted(set(r["n"] for r in subset))
            mean_mses = []
            for n in ns:
                mses = [r["mse"] for r in subset if r["n"] == n]
                mean_mses.append(np.mean(mses))

            ax.loglog(ns, mean_mses, "o-", label="KernelHAR", markersize=4)

            # Reference line: n^{-1/3}
            ns_arr = np.array(ns, dtype=float)
            ref = mean_mses[0] * (ns_arr / ns_arr[0]) ** (-1 / 3)
            ax.loglog(ns, ref, "--", color="gray", alpha=0.6, label=r"$n^{-1/3}$")

            ax.set_title(f"{dgp}, d={d}")
            if i == len(dgps) - 1:
                ax.set_xlabel("n")
            if j == 0:
                ax.set_ylabel("MSE (vs truth)")
            ax.legend(fontsize=7)

    fig.suptitle("Convergence Rate Verification", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "convergence.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_timing(results):
    """Plot training time vs n for each DGP and dimension."""
    import matplotlib.pyplot as plt

    dgps = sorted(set(r["dgp"] for r in results))
    dims = sorted(set(r["d"] for r in results))

    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 3.5), squeeze=False)

    for j, d in enumerate(dims):
        ax = axes[0][j]
        for dgp in dgps:
            subset = [r for r in results if r["dgp"] == dgp and r["d"] == d
                      and r["method"] == "KernelHAR"]
            ns = sorted(set(r["n"] for r in subset))
            mean_times = []
            for n in ns:
                times = [r["train_time"] for r in subset if r["n"] == n]
                mean_times.append(np.mean(times))
            ax.plot(ns, mean_times, "o-", label=dgp, markersize=4)

        ax.set_title(f"d={d}")
        ax.set_xlabel("n")
        if j == 0:
            ax.set_ylabel("Training time (s)")
        ax.legend(fontsize=7)

    fig.suptitle("Training Time vs Sample Size", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "timing.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_benchmarks(results):
    """Bar chart of mean MSE across UCI datasets."""
    import matplotlib.pyplot as plt

    datasets = sorted(set(r["dataset"] for r in results))
    mean_mses = []
    std_mses = []
    for ds in datasets:
        mses = [r["mse"] for r in results if r["dataset"] == ds]
        mean_mses.append(np.mean(mses))
        std_mses.append(np.std(mses))

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 0.8), 4))
    x = np.arange(len(datasets))
    ax.bar(x, mean_mses, yerr=std_mses, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("KernelHAR — UCI Benchmark MSE")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "benchmarks.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    sim_path = os.path.join(RESULTS_DIR, "simulation_results.json")
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim_results = json.load(f)
        print("Plotting simulation results...")
        plot_convergence(sim_results)
        plot_timing(sim_results)
    else:
        print(f"No simulation results found at {sim_path}, skipping.")

    bench_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            bench_results = json.load(f)
        print("Plotting benchmark results...")
        plot_benchmarks(bench_results)
    else:
        print(f"No benchmark results found at {bench_path}, skipping.")


if __name__ == "__main__":
    main()
