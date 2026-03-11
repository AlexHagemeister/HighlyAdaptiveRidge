# Phase 2: Clean Implementations

**Goal:** Consolidate the algorithm implementations into `har/` with consistent, readable interfaces.

**Depends on:** Phase 1 (directory structure exists, dead weight archived)

## Scope

Three files in `har/`:
1. `kernel_har.py` — The kernelized HAR. This is the paper's main contribution.
2. `hal.py` — HAL baseline for comparison experiments.
3. `data_generators.py` — The three DGP families (Smooth, Jump, Sinusoidal).

## Key Decisions

### kernel_har.py
- Source: current `kernel_har.py` (the `KernelHAR` class)
- Keep the kernel computation inline (the `2^|s_i(x,x')|` formula from Section 3.2)
- Pick the best-performing kernel computation approach from `kernel_functions.py` — likely the fully vectorized `compute_K_np` variant. Profile if unsure.
- Clean up: remove print statements, standardize docstrings, ensure predict raises clearly if not fitted
- The sklearn-conforming interface (BaseEstimator/RegressorMixin) is a future task, not this phase. But the API should be fit/predict with numpy arrays in, numpy arrays out.
- CV lambda selection should be clean and configurable (lambda grid, n_folds as params)

### hal.py
- Source: current `highly_adaptive_lasso.py`
- Minimal cleanup: same fit/predict interface as kernel_har, consistent docstring style
- This is a baseline, not the contribution. Don't over-invest.

### data_generators.py
- Source: current `data_generators.py`
- Keep: SmoothDataGenerator, JumpDataGenerator, SinusoidalDataGenerator
- Drop or archive: the legacy `DataGenerator` class (2D only, not used in paper experiments)
- Bug check: `JumpDataGenerator.generate_data` and `SinusoidalDataGenerator.generate_data` both call `SmoothDataGenerator.f_*` for d=1,3,5 instead of their own. Verify whether this was intentional or a copy-paste error. If error, fix.
- Consistent interface: all generators should have `generate_data(n, d, seed=None)` with explicit seeding

## Constraints
- Don't change algorithmic behavior. This is a refactor, not a rewrite.
- If you find a bug, fix it but document what changed and why.
- No new dependencies beyond what's already used (numpy, sklearn, scipy).

## Verify
- [ ] `har/` contains exactly: `__init__.py`, `kernel_har.py`, `hal.py`, `data_generators.py`
- [ ] Each module imports cleanly: `from har.kernel_har import KernelHAR`
- [ ] Quick smoke test: generate data, fit KernelHAR, predict, no crashes
- [ ] No print statements in library code
- [ ] Root-level copies of old files are deleted (they're in `_archive/` from Phase 1)
