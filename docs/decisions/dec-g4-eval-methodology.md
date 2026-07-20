# ADR-G4: Evaluation Methodology (GraphRAG vs Vector Benchmark)

**Date:** 2026-06-11
**Status:** Accepted (judge/aggregation sections extended by ADR-G8)

> **Extension (2026-07-20):** ADR-G8 makes gnomon canonical for judge
> semantics and CI aggregation — `OpenAICompatJudge` remains transport-only
> and the harness's private bootstrap is replaced by gnomon's
> `aggregate_metric`. See `dec-g8-gnomon-as-eval-engine.md`.

## Context

Phase 3 produces the honest number: graph-aware retrieval vs vector baseline vs hybrid,
measured over the same corpus, with confidence intervals. GNOMON has been audited and is
pip-installable; this ADR fixes **how** we measure.

## Decisions

**Query set (P3.2).** 25 authored questions over the Monster Manual graph, frozen in
`eval/queries.json` and versioned. Balanced by hypothesis: relational (single/multi/
entity-relation, where the graph should win) and factual (attribute/description, where vector
should suffice). Each query brings a relevance oracle **derived from the KG** — candidate, not
gold-verified: inherits extraction errors (ex. `ankheg → resists ácido`), kept intentionally
so the report exposes noise (P3.5). `n=25` is declared; expandable.

**Generation step.** GNOMON's `faithfulness` measures how anchored the answer is in the
contexts, so each arm generates an answer over its retrieved context (grounded prompt:
"answer from context only"). Without generation there is no `faithfulness`. Tokens and latency
are real.

**Budget parity.** The three arms truncate context at the same `token_budget` and use the same
local embedder (ADR-G3). The cost differential is the size of context each arm delivers to the
generator (graph segments vs vector chunks) — exactly the token efficiency we measure.

**Judge (P3.0).** GNOMON v1 metrics: **`faithfulness` and `context_precision` only** (answer
relevance and context recall are v2, not built — we do not promise recall). The v1 judge is
**reference-free in practice**: the `EvalCase` schema requires `expected_answer`/`expected_contexts`,
but the judge prompt ignores them; we populate from the query set to validate the schema. We use an
**`OpenAICompatJudge` (Groq, OSS in the cloud)** that reuses GNOMON's prompt and parse — only
the transport changes, so the score means the same as `OllamaJudge`. `seed + run` gives the
sequence deterministic per declared seed.

**Aggregation and CI (P3.3).** Bypass of `run_eval`: it discards per-case scores and we want them
(to report where the graph **lost**, P3.5). We drive the judge per case, collapse the
`judge_runs` by mean (one score per case), and reuse GNOMON's `aggregate_metric` for
**seeded percentile bootstrap** (2000 resamples, `n ≥ 2`). Same CI machine, with per-case detail.

**Cost.** GNOMON only counts tokens; GLYPH calculates USD from **generation** tokens, at
Haiku 4.5's asymmetric rates ($1/M input, $5/M output). The OSS judge is priced separately and
excluded from the table (declared in `METRICS.md`).

**Reproducible baseline (P3.4).** `make benchmark` regenerates `eval/benchmark-baseline.json`
(committed) + `METRICS.md`. `make benchmark-check` fails if a new run diverges beyond the
tolerance (default 0.05) from committed means. Corpus and query set frozen.

**Honest report (P3.5).** The table shows each metric with CI for **all** arms —
including where the graph did not win — plus tokens, cost, and latency. Declares `n`, that tokens are
real count, and the caveat of the KG-derived oracle.

## Consequences

Controlled and reproducible comparison of the repo. The number is not published without running
`make benchmark` with real keys (`GROQ_API_KEY` + `ANTHROPIC_API_KEY`); until then `METRICS.md`
declares it pending. Declared limitations: KG-derived oracle not gold-verified, only two v1 metrics,
OSS judge non-deterministic within measured variance.
