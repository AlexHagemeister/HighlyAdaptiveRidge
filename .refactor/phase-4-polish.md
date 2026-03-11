# Phase 4: README & Polish

**Goal:** Write the final README, lock dependencies, audit .gitignore, and do a full dry-run.

**Depends on:** Phase 3 (experiments run and produce results)

## Scope

- Rewrite `README.md` to match the finished repo
- Create `requirements.txt` with pinned versions
- Final .gitignore audit
- Clean up `.refactor/` (archive or delete — it served its purpose)
- Full dry-run: fresh clone → install deps → run experiments → check figures

## README Structure (target)
1. Title + one-line description
2. Paper reference + link (arXiv or journal)
3. What HAR is (2-3 sentences, accessible)
4. Key result (the convergence rate, why it matters)
5. Quick start (clone, install, reproduce)
6. Repo structure (brief)
7. Citation (BibTeX)
8. License (if applicable)

## Constraints
- README should be readable by someone who hasn't read the paper
- Don't oversell. State what the code does, link to the paper for theory.
- Keep it under ~100 lines of markdown

## Verify
- [ ] Fresh `git clone` + `pip install -r requirements.txt` + run experiments works
- [ ] README accurately describes what's in the repo
- [ ] No leftover artifacts from the refactor (no empty dirs, no stale references)
- [ ] `git log` tells a clean story (phase commits are clear)
