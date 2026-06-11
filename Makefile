# GLYPH developer + benchmark tasks.
#
# The benchmark needs API keys and is not run on every push (see docs/GLYPH_PLAN.md
# Phase 3). Provide the source book PDF via BOOK= (it is not committed).

GRAPH ?= out/monster-manual.json
BOOK ?=
JUDGE_MODEL ?= llama-3.3-70b-versatile

.PHONY: test lint typecheck check query-set query-set-check benchmark benchmark-check

test:
	pytest

lint:
	ruff check glyph tests
	ruff format --check glyph tests

typecheck:
	mypy glyph

check: lint typecheck test

query-set:
	python3 scripts/build_query_set.py

query-set-check:
	python3 scripts/build_query_set.py --check

# Real run: regenerates eval/benchmark-baseline.json + METRICS.md.
# Requires GROQ_API_KEY and ANTHROPIC_API_KEY in the environment.
benchmark:
	@test -n "$(BOOK)" || (echo "set BOOK=<path to the source PDF>"; exit 2)
	python3 scripts/run_benchmark.py $(GRAPH) "$(BOOK)" --model $(JUDGE_MODEL)

# Regression gate: fail if a fresh run drifts past tolerance from the committed baseline.
benchmark-check:
	@test -n "$(BOOK)" || (echo "set BOOK=<path to the source PDF>"; exit 2)
	python3 scripts/run_benchmark.py $(GRAPH) "$(BOOK)" --model $(JUDGE_MODEL) --check
