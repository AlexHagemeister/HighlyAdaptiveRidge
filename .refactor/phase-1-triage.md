# Phase 1: Triage & Secure

**Goal:** Remove security risks, archive non-essential files, establish the target directory skeleton.

After this phase the repo should have the right *shape* even if the contents aren't clean yet.

## Checklist

### Security — ON MAIN, BEFORE BRANCHING
The keyfiles must be purged from git history before creating the refactor branch, otherwise the branch inherits dirty history and you'd have to rebase after filtering.

- [x] Ensure working tree is clean (`git status`)
- [x] Delete `keyfile` and `keyfile.pub` from the working tree
- [x] Install `git-filter-repo` if needed (`pip install git-filter-repo` or `brew install git-filter-repo`)
- [x] Purge from history: `git filter-repo --invert-paths --path keyfile --path keyfile.pub --force`
- [x] **Re-add the remote** — `git filter-repo` strips remote config by design as a safety measure: `git remote add origin https://github.com/AlexHagemeister/HighlyAdaptiveRidge`
- [x] Verify: `git log --all --full-history -- keyfile` should return nothing
- [x] Force-push: `git push origin main --force`
- [ ] Verify keys are rotated / no longer in use anywhere (ask user if unsure)

### Create the refactor branch (after security cleanup)
- [x] `git checkout -b refactor/cleanup-for-publication`

### Directory structure
- [x] Create `har/` with `__init__.py`
- [x] Create `experiments/`
- [x] Create `results/figures/`
- [x] Rename `csv/` → `data/` (use `git mv`) — csv/ was gitignored, used plain mv
- [x] Create `_archive/`

### Archive dead weight
Move the following to `_archive/` (use `git mv` where possible):
- [x] `notebooks/` (both ipynb files)
- [x] `find_best_kernel.ipynb`
- [x] `.ipynb_checkpoints/`
- [x] `simulation_1.0.py` (superseded by `run_trials.py`)
- [x] `kernel_functions.py` (optimization exploration — best version goes into kernel_har.py in Phase 2)
- [x] `df_pickles/` (generated artifacts, not source)
- [x] `plots/` (will be regenerated in Phase 3)
- [x] `models/` (empty besides `__init__.py` and `__pycache__`)
- [x] `env_notes.txt`
- [x] `train_time_plotter.py` (Altair plotter — will be rewritten in Phase 3)
- [x] `.vscode/`
- [x] `Pipfile` and `Pipfile.lock` (replacing with requirements.txt in Phase 4)

### Gitignore
- [x] Add `_archive/` to `.gitignore`
- [x] Add `results/` to `.gitignore` (except maybe `results/.gitkeep`)
- [x] Add `__pycache__/`, `*.pyc`, `.DS_Store`, `.ipynb_checkpoints/` — already present in existing gitignore
- [x] Remove any existing entries that no longer apply

### Verify
- [x] `git log --all --full-history -- keyfile` returns nothing
- [x] `git status` shows only the structural changes, no secrets
- [x] Repo root is clean: README.md, har/, experiments/, data/, results/, _archive/, .refactor/, .gitignore
- [ ] Commit with message like `refactor(phase-1): triage, secure, restructure`

## Notes
- Don't refactor any code yet. Just move files. Phase 2 handles the code.
- The security steps happen on `main` and are force-pushed. Everything after happens on the refactor branch.
- If `keyfile`/`keyfile.pub` were ever pushed to a public remote, treat them as compromised regardless of whether they "look" important.
- `git filter-repo` rewrites all commit hashes. This is expected and fine since the repo is private.
- `git filter-repo` also strips the remote config as a safety measure. The checklist includes re-adding it.
