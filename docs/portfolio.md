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

## From the benchmark runs (two domains, honest)

> Benchmarked graph-aware vs vector vs hybrid retrieval across **two real corpora** with
> bootstrap CIs and independent LLM judges. On **documents** (Monster Manual, n=25) graph-aware
> retrieval led faithfulness (**0.987** [0.96–1.00]) at the **lowest token cost**, parity-or-better
> with vector on both metrics. On **code** (AXON source graph, n=14, independent Gemini judge) the
> result inverted: the **fair vector baseline beat graph on both metrics** (faithfulness 0.995 vs
> 0.839, context_precision 0.513 vs 0.180) — the graph's hop-expansion added noise. Reported with
> CIs, including where graph lost and a documented judge self-evaluation bias.

The credibility is the method, not a single number: a comparison built to let either side win,
two domains with opposite outcomes, both reported. Numbers from `eval/benchmark-baseline.json`
/ `METRICS.md` (documents) and `eval/code-benchmark-baseline.json` / `METRICS-code.md` (code).

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
