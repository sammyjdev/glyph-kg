# Portfolio claims (P6.3 / P6.4)

Validation-first: a claim ships only when the repo backs it. Two tiers below — what is
provable **today** from the code, and what unlocks **after** a real benchmark run.

## Provable today (machinery + artifacts)

> Built GLYPH, a knowledge-graph library spanning **document** (LLM extraction) and **code**
> (tree-sitter, Python + Java) domains behind one extractor port, with graph-aware retrieval,
> a fair vector baseline, and a reproducible benchmark harness (GNOMON, bootstrap confidence
> intervals). Hexagonal architecture, invariants enforced in CI, 100% test coverage.

Backed by: `glyph/` (model, store, extract/document, extract/code, retrieval, baseline, eval,
integration), `out/monster-manual.json`, `out/glyph-code.json`, ADRs G1–G6, CI quality gates.

## From the benchmark run (the headline number)

> Benchmarked graph-aware vs vector retrieval over a real 25-query corpus with bootstrap CIs
> (judge: Llama 3.3 70B). Graph-aware retrieval posted the **highest faithfulness (0.987,
> [0.96–1.00])** at the **lowest token cost (~12% fewer tokens than the vector baseline)**,
> matching or beating vector on both v1 metrics. At n=25 the metric CIs overlap, so the honest
> framing is *parity-or-better on quality at lower cost* — not a significance claim; the graph
> did **not** lead `context_precision` (vector/hybrid nominally higher, within noise).

Numbers from `eval/benchmark-baseline.json` / `METRICS.md`. The credibility is in reporting the
overlap and the category where the graph lost, not in inflating a single number.

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
