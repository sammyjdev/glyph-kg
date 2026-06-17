# Portfolio claims (P6.3 / P6.4)

Validation-first: a claim ships only when the repo backs it. Two tiers below — what is
provable **today** from the code, and what unlocks **after** a real benchmark run.

## Provable today (machinery + artifacts)

> Built GLYPH, a knowledge-graph library spanning **document** (LLM extraction) and **code**
> (tree-sitter, Python + Java) domains behind one extractor port, with graph-aware retrieval,
> a fair vector baseline, and a reproducible benchmark harness (GNOMON, bootstrap confidence
> intervals). Hexagonal architecture, invariants enforced in CI, ~98% test coverage (≥90% CI gate).

Backed by: `glyph/` (model, store, extract/document, extract/code, retrieval, baseline, eval,
integration), `out/monster-manual.json`, `out/glyph-code.json`, ADRs G1–G7, CI quality gates.

## From the benchmark runs (three axes, honest)

> Benchmarked graph-aware vs vector vs hybrid retrieval across **three setups** with bootstrap CIs.
> On **documents** (Monster Manual, n=25) graph-aware retrieval led faithfulness (**0.987**
> [0.96–1.00]) at the **lowest token cost**, parity-or-better with vector. On **code** (AXON source
> graph, n=14), judged by **two independent models** (Gemini + Qwen) to test robustness: the fair
> **vector baseline led faithfulness under both judges** (vector > hybrid > graph, consistently),
> while `context_precision` proved **judge-dependent** (ranking inverted between judges) — so I report
> no winner on it. On the **global community axis** (GraphRAG-style, n=8) community summaries gave
> parity-or-better answers at **~half the tokens** of vector/graph — *same answer, half the cost* for
> "how is this organized?" questions. Cross-family judging turned clean single-judge results into
> correctly hedged ones.

The credibility is the method, not a single number: comparisons built to let either side win, three
setups, a robust metric separated from a non-robust one, and a single-judge conclusion overturned by
a second independent judge — all reported. Sources: `eval/benchmark-baseline.json` (documents),
`eval/code-benchmark-baseline.json` + `eval/code-benchmark-qwen.json` (code, two judges),
`eval/code-global-baseline.json` + `eval/code-global-qwen.json` (global axis, two judges).

## CV / LinkedIn one-liner

> "Built a knowledge-graph library spanning document and code domains; benchmarked graph-aware
> vs vector retrieval with confidence intervals."

Safe to use today — the run has landed (`eval/benchmark-baseline.json`); the comparative number is real.

## Visibility plan (P6.4)

- Showcase in the Qdrant and MCP Discords; put the reproducible-benchmark link in the **first
  comment**, not the post body.
- Lead with the honest table (CIs, including where the graph lost) — the credibility is in the
  limitations, not in hiding them.
- Link `docs/article.md` (method) and `METRICS.md` (numbers) so a reader can re-run it.
