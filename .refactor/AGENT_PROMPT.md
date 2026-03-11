# HAR Repo Refactor — Agent Prompt

## Context

You are refactoring a research repository for a published ML paper (Highly Adaptive Ridge). The refactor is planned in `.refactor/REFACTOR.md` (the root plan) with detailed phase docs linked from it.

**`HAR_Paper.md` in the project root is a markdown version of the published paper.** This is your primary reference for understanding the algorithm, the kernel trick, the convergence result, and what experiments the code needs to reproduce. Read it early. Return to it whenever you need to decide what matters.

## Setup

```
cd /Users/alexhagemeister/dev/HAR/Highly_Adaptive_Ridge
gh auth status  # confirm you're authed
```

**Do NOT create a branch yet.** Phase 1 begins with security cleanup steps that must happen on `main` before branching. The phase doc specifies when to branch.

## How to work

1. **Read the root plan first.** Open `.refactor/REFACTOR.md`. Understand the goal, target structure, key decisions, and which phase is next (the first unchecked box).

2. **Read the current phase doc.** Follow the link from REFACTOR.md to the detailed phase file (e.g. `.refactor/phase-1-triage.md`).

3. **Before executing: verify the plan against reality.** The plan was written by a human+AI pair who could see the repo but may have made errors or assumptions. For every claim in the phase doc:
   - Check that referenced files actually exist where the plan says they are.
   - Check that files contain what the plan assumes they contain.
   - If something doesn't match, adapt. Note what diverged and why in a brief comment at the bottom of the phase doc under a `## Deviations` section.

4. **Execute the phase.** Work through the checklist in order. For each item:
   - Do the thing.
   - Verify it worked (ls, git status, quick test — whatever's appropriate).
   - Check off the item in the phase doc (change `- [ ]` to `- [x]`).

5. **After completing a phase:**
   - Run the verification steps at the bottom of the phase doc.
   - Commit with the suggested message format.
   - Update REFACTOR.md: check off the completed phase.
   - Skim the *next* phase doc and note if anything needs to change based on what you just did. If so, edit it. Don't start executing the next phase — just adjust the plan.

## Principles

- **The paper is the source of truth.** `HAR_Paper.md` tells you what the algorithm does, what the kernel formula is, what experiments matter. When in doubt, read the paper.
- **Trust but verify.** The plan is a guide, not a script. If the plan says "move X to Y" but X doesn't exist, figure out where it actually is (or whether it matters) before proceeding.
- **No code refactoring in Phase 1.** Phase 1 is purely structural: delete secrets, move files, create directories. Code changes happen in Phase 2+.
- **Prefer git operations.** Use `git mv` instead of plain `mv` so history is preserved.
- **Security first.** The keyfile purge from git history is non-negotiable. It happens on `main` before branching. Use `git filter-repo` (install if needed). The repo is private and has been pushed to origin — filter-repo will rewrite history, which is expected and fine. Note: filter-repo strips remote config; the phase doc includes the step to re-add it.
- **Ask me if genuinely stuck.** But exhaust what you can figure out from the codebase and the paper first.

## Start

Begin with Phase 1. Read `.refactor/phase-1-triage.md`, verify its assumptions against the actual repo state, then execute. Remember: security steps first, on `main`, then branch, then the rest of the phase.
