## Validation: issue #11 — PASS
Spec-anchored check: no spec.md for this issue (entered via `task` directly) — fallback: assertion exists and covers each criterion (3/3 ACs matched: config exists + hooks configured, install documented, scope-only AC3 exempt).
Mutation sensor (mandatory): EMPTY_RETURN=KILLED, IDENTITY_RETURN=KILLED, NEGATE_CONDITIONAL=KILLED, DROP_SIDE_EFFECT=KILLED
Mutation sensor (extras): N/A: Common tier, no extras required
Report: .specs/features/issue-11-pre-commit-config/validation.md

Notes: this is a config+docs deliverable (no application return values/conditionals), so the four mandatory operators were run as behavior-level analogs against the two artifacts (.pre-commit-config.yaml, CONTRIBUTING.md) rather than literal code mutations:
- EMPTY_RETURN analog: truncated .pre-commit-config.yaml to empty -> test_ruff_hook_configured + test_gitleaks_hook_configured failed (KILLED).
- IDENTITY_RETURN analog: dropped the gitleaks hook block entirely (as if the maker only did half the work) -> test_gitleaks_hook_configured failed (KILLED).
- NEGATE_CONDITIONAL analog: mangled the documented command (`pre-commit install` -> `pre-commit installed-hooks`) -> test_pre_commit_install_is_documented failed (KILLED). This SURVIVED on the first pass (substring match false-positive) and was sent back to the maker for a fix (exact-line-match assertion); re-verified KILLED after the fix.
- DROP_SIDE_EFFECT analog: reverted CONTRIBUTING.md's new doc section entirely -> test_pre_commit_install_is_documented failed (KILLED).

All scratch mutations were applied to temp copies / reverted immediately after each check; the worktree diff was confirmed byte-identical to pre-mutation state after each step (`diff` + `git diff`).
