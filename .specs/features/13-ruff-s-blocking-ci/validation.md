## Validation: issue #13 — PASS
Spec-anchored check: N/A (no spec.md; issue entered `task` directly) — fallback: assertion exists and covers each criterion (4/4 ACs matched: S in select, redundant CI step removed, S101 comment updated, ruff check green).
Mutation sensor (mandatory): EMPTY_RETURN=KILLED, IDENTITY_RETURN=KILLED, NEGATE_CONDITIONAL=N/A: pure config/CI deliverable, no conditional logic to invert, DROP_SIDE_EFFECT=KILLED
Mutation sensor (extras): N/A (Common tier — no extras required)
Report: .specs/features/13-ruff-s-blocking-ci/validation.md

### Mutation detail
- EMPTY_RETURN: dropped `"S"` from `[tool.ruff.lint] select` -> `test_s_is_in_default_ruff_selection` and `test_default_ruff_selection_blocks_an_s_finding` fail.
- IDENTITY_RETURN: restored the old redundant `ruff check glyph tests --select S --exit-zero` CI step -> `test_ci_has_only_the_main_blocking_ruff_check` fails.
- DROP_SIDE_EFFECT: removed `S101` from `per-file-ignores["tests/**"]` -> `ruff check` produces 604 real S101 findings in `tests/`; `test_s101_remains_ignored_only_for_pytest_files` and `test_default_ruff_check_is_green` fail.
- All mutations discarded immediately after checking; the worktree was restored to the correct diff and re-verified green (`make check` exit 0, 353 passed, 96.91% coverage) before this report was written.
