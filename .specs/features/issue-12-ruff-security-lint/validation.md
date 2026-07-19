## Validation: issue #12 — PASS
Spec-anchored check: no spec.md for this issue (entered via `task` directly) — fallback: assertion exists and covers each criterion (CI report-only S-lint step: verified manually, `ruff check glyph tests --select S --exit-zero` -> 0 findings; S310 in judge.py, S112 in extractor.py, S324 in community.py: each has a dedicated test; S101 per-file-ignore: config-only, verified manually).
Mutation sensor (mandatory): EMPTY_RETURN=KILLED, IDENTITY_RETURN=KILLED, NEGATE_CONDITIONAL=KILLED, DROP_SIDE_EFFECT=KILLED
Mutation sensor (extras): 2 injected, 2 killed (after 1 fix round): BOUNDARY (drop "https" from _ALLOWED_SCHEMES allow-list) SURVIVED on first pass -> fixed by adding a monkeypatched-urlopen accept-path test, re-verified KILLED; EXCEPTION_SWALLOW (narrow extractor.py's except clause from Exception to ValueError) KILLED on first pass.
Report: .specs/features/issue-12-ruff-security-lint/validation.md

Notes on operator mapping (config/docs pieces are non-code, most real logic is guard-clauses/logging, not classic return-value transforms):
- IDENTITY_RETURN analog: deleted the judge.py scheme guard entirely -> test_urllib_post_rejects_non_http_scheme_before_opening failed (KILLED; the file:// read succeeded locally and json.loads raised JSONDecodeError, a ValueError subtype, but with the wrong message, so pytest.raises(match=...) still failed correctly).
- NEGATE_CONDITIONAL: inverted `scheme not in _ALLOWED_SCHEMES` to `scheme in _ALLOWED_SCHEMES` -> same test KILLED.
- EMPTY_RETURN analog: downgraded extractor.py's logger.warning to logger.info (caplog.at_level(WARNING) misses it) -> test_extract_logs_a_warning_when_a_chunk_raises failed (KILLED).
- DROP_SIDE_EFFECT: removed the logger.warning call before the first `continue` in extractor.py -> same test failed (KILLED).
- BOUNDARY (Rare-tier extra): shrank judge.py's allow-list from {http,https} to {http} -> SURVIVED first pass (no positive-path test existed for a valid scheme); sent back to maker, fixed with a monkeypatched-urlopen positive test, re-verified KILLED.
- EXCEPTION_SWALLOW (Rare-tier extra): narrowed extractor.py's `except Exception` to `except ValueError` -> the test's injected RuntimeError propagated uncaught -> test errored (KILLED).

community.py's `usedforsecurity=False` change is a byte-identical annotation (verified: existing tests/retrieval/test_community.py suite unchanged and still green) — no mutation applicable, no side effect to drop.

Review fix round 1/2: spec-compliance+quality reviewer (Rare tier, Sonnet 5) found extractor.py's SECOND logger.warning (notes-schema coercion except-block) had no test or exemption. Fixed with a test-only addition (`test_extract_logs_a_warning_when_notes_coercion_fails` in tests/extract/document/test_extractor_logging.py) — no production code touched. Re-verified via comment-out-line mutation check (fails when that line is removed) and a scoped re-review: APPROVE.

All scratch mutations were applied directly to the two real source files and reverted immediately after each check; `git diff`/`diff` against the maker's committed-in-this-branch (uncommitted) state confirmed byte-identical restoration after every mutation.
