# Portfolio claims (P6.3 / P6.4)

Validation-first: a claim ships only when the repo backs it. Two tiers below — what is
provable **today** from the code, and what unlocks **after** a real benchmark run.

## Provable today (machinery + artifacts)

> Built GLYPH, a knowledge-graph library spanning **document** (LLM extraction) and **code**
> (tree-sitter, Python + Java) domains behind one extractor port, with graph-aware retrieval,
> a fair vector baseline, and a reproducible benchmark harness (GNOMON, bootstrap confidence
> intervals). Hexagonal architecture, invariants enforced in CI, 100% test coverage.

Backed by: `glyph/` (model, store, extract/document, extract/code, retrieval, baseline, eval,
integration), `out/monster-manual.json`, `out/glyph-code.json`, ADRs G1–G5, CI quality gates.

## Unlocks after the benchmark run (the headline number)

> Benchmarked graph-aware vs vector retrieval over a real corpus with confidence intervals:
> graph improved `context_precision` by **[X] pts [95% CI a–b]** on relation-dependent queries
> at **[Y]% of the token cost**, while the vector baseline matched it on simple factual lookups.

Fill `[X]/[Y]` from `METRICS.md` after `make benchmark`. **Do not publish the number before the
run** — the validation-first rule. Report the categories where the graph did **not** win too.

## CV / LinkedIn one-liner

> "Built a knowledge-graph library spanning document and code domains; benchmarked graph-aware
> vs vector retrieval with confidence intervals."

Safe to use once the run lands (the capability is built today; the comparative number is the run's).

## Visibility plan (P6.4)

- Showcase in the Qdrant and MCP Discords; put the reproducible-benchmark link in the **first
  comment**, not the post body.
- Lead with the honest table (CIs, including where the graph lost) — the credibility is in the
  limitations, not in hiding them.
- Link `docs/article.md` (method) and `METRICS.md` (numbers) so a reader can re-run it.
