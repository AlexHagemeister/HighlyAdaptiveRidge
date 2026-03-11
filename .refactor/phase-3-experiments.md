# Phase 3: Experiments & Reproducibility

**Goal:** Build scripts in `experiments/` that reproduce the paper's empirical results.

**Depends on:** Phase 2 (clean implementations in `har/`)

## Scope

The paper's Section 4 (Demonstration) covers:
1. **Convergence rate verification** — MSE vs. n on simulated data, confirming the n^{-1/3} rate
2. **Timing comparison** — Training time: HAR vs HAL (and optionally GBT, KRR with fixed kernel)
3. **Real-data benchmarks** — UCI regression datasets (the CSVs in `data/`)

## Intent

Build 2-3 scripts that are runnable end-to-end:
- `run_simulations.py` — Convergence + timing on synthetic DGPs across dimensions and sample sizes
- `run_benchmarks.py` — Evaluate on UCI datasets
- `plot_results.py` — Load saved results, generate publication-quality figures to `results/figures/`

## Design Constraints
- All experiments seeded for reproducibility
- Results saved as CSV or JSON (not pickle) so they're inspectable
- Expected runtimes documented (comment at top of each script)
- Figures should be close to what's in the paper (or better)
- Should be runnable as: `python experiments/run_simulations.py` from repo root

## Open Questions (resolve at start of this phase)
- Which UCI datasets were actually used in the paper's final results? (The `data/` folder has ~12 CSVs — may not all be needed)
- Does the paper include GBT/KRR-fixed-kernel baselines in the final version, or just HAR vs HAL?
- What sample sizes and dimensions were used in the simulations?

## Verify
- [ ] `python experiments/run_simulations.py` runs to completion and saves results
- [ ] `python experiments/run_benchmarks.py` runs to completion and saves results
- [ ] `python experiments/plot_results.py` generates figures in `results/figures/`
- [ ] Figures visually match the paper's claims (HAR faster than HAL, similar MSE, correct rate scaling)
