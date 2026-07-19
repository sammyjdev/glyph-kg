# Contributing to GLYPH

## Methodology

GLYPH follows Spec-Driven Development: the spec and quality gates are defined before implementation. Agents execute under architectural direction; new architectural decisions become ADRs before becoming code.

## Workflow

1. Pick a sub-task from `docs/GLYPH_PLAN.md` (or the current phase KICKOFF).
2. Write the test before implementation (TDD).
3. Implement until the test passes.
4. Ensure lint, types, coverage, and architecture invariants are green locally.
5. Open a PR referencing the sub-task.

## Quality gates (enforced in CI)

- **Tests:** `pytest` green. Nothing merges without tests.
- **Coverage:** gate active; PRs that lower coverage fail.
- **Types:** `mypy` without error.
- **Lint:** `ruff` clean.
- **Architecture:** `tests/architecture/` verifies invariants from `docs/ARCHITECTURE.md`. Import rule violations fail the build.

## Pre-commit hooks

`.pre-commit-config.yaml` runs `ruff` (lint) and `gitleaks` (secret scan) on staged changes. One-time setup after cloning:

```bash
pip install pre-commit
pre-commit install
```

## ADR rule

Every architectural decision is recorded in `docs/decisions/` before implementation. If an unforeseen decision arises during execution (library choice, contract change, new backend), stop and open an ADR. Do not decide inline in code.

Format: see existing ADRs (`dec-g1-...`). Status, Context, Decision, Consequences (positive / trade-offs / to monitor), Alternatives considered.

## Claim honesty

- README and docs are the source of truth. Do not assert capability or numbers that the code does not deliver.
- Published metrics are reproducible from the repo. Without reproduction, do not publish.
- Known limitations are documented, not hidden.

## Commits

- Message in imperative mood, small scope per sub-task.
- One PR solves one sub-task or a coherent set.

## Merge / integration

- Branches integrate into `main` via **rebase + fast-forward** — linear history, no merge commits.
  Workflow: `git rebase main` on the branch, then `git checkout main && git merge --ff-only <branch>`.
- On GitHub, the only enabled method is **Rebase and merge** (merge commits and squash disabled).
- `git config pull.rebase true` keeps pulls linear as well.

## Style

- Active voice, dense prose, no fluff.
- No "robust", "scalable", "powerful" without concrete evidence.
- Vague metrics ("improves performance") are removed or gain a number.
