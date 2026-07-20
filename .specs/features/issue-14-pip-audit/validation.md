## Validation: issue #14 — PASS
Spec-anchored check: no spec.md for this issue (entered via `task` directly) — fallback: assertion exists and covers each criterion (pip-audit step present, upgrade-before-audit ordering, blocking invocation, not a project dependency: all 4 asserted; "job green on GHA" exempt, verified manually instead — see below).
Mutation sensor (mandatory): EMPTY_RETURN=KILLED, IDENTITY_RETURN=KILLED, NEGATE_CONDITIONAL=KILLED, DROP_SIDE_EFFECT=KILLED
Mutation sensor (extras): 2 injected, 2 killed: EXCEPTION_SWALLOW (`|| true` suffix) KILLED, BOUNDARY/scope (add pip-audit to pyproject.toml dev deps) KILLED
Report: .specs/features/issue-14-pip-audit/validation.md

Notes on operator mapping (CI-YAML-only change, no application code):
- EMPTY_RETURN analog: removed both the pip/setuptools-upgrade and pip-audit steps entirely -> all 4 structural tests failed (KILLED).
- IDENTITY_RETURN analog: kept `pip install pip-audit` but dropped the actual `pip-audit` invocation line -> 3/4 tests failed (KILLED).
- NEGATE_CONDITIONAL analog: reordered steps so pip-audit runs BEFORE the pip/setuptools upgrade -> test_pip_upgrade_before_pip_audit failed (KILLED).
- DROP_SIDE_EFFECT (mandatory, mapped to "blocking" property): appended `--exit-zero` to the pip-audit line -> test_pip_audit_is_blocking failed (KILLED).
- EXCEPTION_SWALLOW (extra): appended `|| true` instead -> same test failed (KILLED).
- BOUNDARY/scope (extra): added "pip-audit" as a pyproject.toml dev dependency (simulating scope creep from CI-only tool to project dependency) -> test_pip_audit_installed failed (KILLED).

Manual verification (the one exempt criterion): built a clean scratch venv (not this worktree's `.venv`), ran `pip install --upgrade pip setuptools && pip install -e '.[dev,document,retrieval,code,eval]' pip-audit && pip-audit` -> "No known vulnerabilities found" (0 findings). Confirms the CI job as written will pass on GitHub Actions with the current dependency set.

All scratch mutations were applied directly to ci.yml/pyproject.toml and reverted immediately after each check; `diff` against the pre-mutation copies confirmed byte-identical restoration after every mutation.
