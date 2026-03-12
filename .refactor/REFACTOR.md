# HAR Repository Refactor

## Goal

Transform this research repo into a clean, reproducible companion to the published paper:

> **Highly Adaptive Ridge** — Schuler, Hagemeister, van der Laan (2024)
> *n^{-1/3} dimension-free L2 convergence for kernel ridge regression with a data-adaptive zero-order spline basis.*

The end state: someone reads the paper, clones this repo, runs a script, and gets the results. No archaeology required.

## Primary Reference

**`HAR_Paper.md`** in the project root is a markdown version of the published paper. This is the source of truth for what the algorithm does, what experiments matter, and what claims the code needs to support. When in doubt about what to keep, cut, or how something should work — check the paper.

## Target Structure

```
Highly_Adaptive_Ridge/
├── README.md                     # paper summary, quick start, citation
├── HAR_Paper.md                  # markdown version of the published paper (reference)
├── har/                          # clean implementations
│   ├── __init__.py
│   ├── kernel_har.py             # kernelized HAR (the paper's contribution)
│   ├── hal.py                    # HAL baseline
│   └── data_generators.py        # Smooth, Jump, Sinusoidal DGPs
├── experiments/
│   ├── run_simulations.py        # convergence rate + timing experiments
│   ├── run_benchmarks.py         # UCI real-data benchmarks
│   └── plot_results.py           # figure generation
├── data/                         # UCI benchmark CSVs
│   └── *.csv
├── results/
│   └── figures/                  # generated plots
├── requirements.txt
├── .gitignore
└── _archive/                     # old notebooks, scratch (gitignored)
```

## Key Decisions

- **Kernel HAR only.** The explicit-basis implementation doesn't scale (the paper's Section 3.2 is about avoiding it). Archive it.
- **No package infrastructure yet.** No setup.py/pyproject.toml. That's a separate future effort (sklearn-conforming API, pip-installable).
- **One good kernel implementation.** The six variants in `kernel_functions.py` collapse into the best one inside `kernel_har.py`.
- **Simulation scripts are deterministic.** Seeded RNG, pinned dependencies, documented expected runtimes.
- **SSH keys must be purged from git history**, not just deleted.

## Phases

Each phase has a summary here and a detailed work document linked below. Work through phases in order. After completing a phase, review downstream phase docs and adjust if needed before proceeding.

- [x] **Phase 1: Triage & secure** — Remove secrets, archive dead weight, establish directory structure.
  → [.refactor/phase-1-triage.md](.refactor/phase-1-triage.md)

- [x] **Phase 2: Clean implementations** — Consolidate HAR, HAL, and DGP code into `har/` with consistent interfaces.
  → [.refactor/phase-2-implementations.md](.refactor/phase-2-implementations.md)

- [x] **Phase 3: Experiments & reproducibility** — Build `experiments/` scripts that produce the paper's simulation and benchmark results.
  → [.refactor/phase-3-experiments.md](.refactor/phase-3-experiments.md)

- [ ] **Phase 4: README & polish** — Final README, requirements.txt, .gitignore audit, dry-run the full reproduce flow.
  → [.refactor/phase-4-polish.md](.refactor/phase-4-polish.md)

## Working Principles

- The README is the goal description. The code should match what it claims.
- **The paper (`HAR_Paper.md`) is the source of truth.** Reference it when making decisions about what the code should do, which experiments to reproduce, and what the kernel formula should be.
- Prefer deleting code over commenting it out. Git has history.
- If something is ambiguous, check the paper first.
- Don't gold-plate. This is a research companion, not a production library.
