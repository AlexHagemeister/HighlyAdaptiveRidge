# Phase 1: Triage & Secure

**Goal:** Remove security risks, archive non-essential files, establish the target directory skeleton.

After this phase the repo should have the right *shape* even if the contents aren't clean yet.

## Checklist

### Security — ON MAIN, BEFORE BRANCHING
The keyfiles must be purged from git history before creating the refactor branch, otherwise the branch inherits dirty history and you'd have to rebase after filtering.

- [ ] Ensure working tree is clean (`git status`)
- [ ] Delete `keyfile` and `keyfile.pub` from the working tree
- [ ] Install `git-filter-repo` if needed (`pip install git-filter-repo` or `brew install git-filter-repo`)
- [ ] Purge from history: `git filter-repo --invert-paths --path keyfile --path keyfile.pub --force`
- [ ] **Re-add the remote** — `git filter-repo` strips remote config by design as a safety measure: `git remote add origin https://github.com/AlexHagemeister/HighlyAdaptiveRidge`
- [ ] Verify: `git log --all --full-history -- keyfile` should return nothing
- [ ] Force-push: `git push origin main --force`
- [ ] Verify keys are rotated / no longer in use anywhere (ask user if unsure)

### Create the refactor branch (after security cleanup)
- [ ] `git checkout -b refactor/cleanup-for-publication`

### Directory structure
- [ ] Create `har/` with `__init__.py`
- [ ] Create `experiments/`
- [ ] Create `results/figures/`
- [ ] Rename `csv/` → `data/` (use `git mv`)
- [ ] Create `_archive/`

### Archive dead weight
Move the following to `_archive/` (use `git mv` where possible):
- [ ] `notebooks/` (both ipynb files)
- [ ] `find_best_kernel.ipynb`
- [ ] `.ipynb_checkpoints/`
- [ ] `simulation_1.0.py` (superseded by `run_trials.py`)
- [ ] `kernel_functions.py` (optimization exploration — best version goes into kernel_har.py in Phase 2)
- [ ] `df_pickles/` (generated artifacts, not source)
- [ ] `plots/` (will be regenerated in Phase 3)
- [ ] `models/` (empty besides `__init__.py` and `__pycache__`)
- [ ] `env_notes.txt`
- [ ] `train_time_plotter.py` (Altair plotter — will be rewritten in Phase 3)
- [ ] `.vscode/`
- [ ] `Pipfile` and `Pipfile.lock` (replacing with requirements.txt in Phase 4)

### Gitignore
- [ ] Add `_archive/` to `.gitignore`
- [ ] Add `results/` to `.gitignore` (except maybe `results/.gitkeep`)
- [ ] Add `__pycache__/`, `*.pyc`, `.DS_Store`, `.ipynb_checkpoints/`
- [ ] Remove any existing entries that no longer apply

### Verify
- [ ] `git log --all --full-history -- keyfile` returns nothing
- [ ] `git status` shows only the structural changes, no secrets
- [ ] Repo root is clean: README.md, har/, experiments/, data/, results/, _archive/, .refactor/, .gitignore
- [ ] Commit with message like `refactor(phase-1): triage, secure, restructure`

## Notes
- Don't refactor any code yet. Just move files. Phase 2 handles the code.
- The security steps happen on `main` and are force-pushed. Everything after happens on the refactor branch.
- If `keyfile`/`keyfile.pub` were ever pushed to a public remote, treat them as compromised regardless of whether they "look" important.
- `git filter-repo` rewrites all commit hashes. This is expected and fine since the repo is private.
- `git filter-repo` also strips the remote config as a safety measure. The checklist includes re-adding it.
