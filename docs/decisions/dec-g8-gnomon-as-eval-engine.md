# ADR-G8: GNOMON Remains the Evaluation Engine; Own Judge Is Transport Only

**Date:** 2026-07-20
**Status:** Accepted (2026-07-20)

## Context

ADR-G4 introduced `OpenAICompatJudge` (`glyph/eval/judge.py:90-134`) as a
sister implementation of GNOMON's judge: same prompt and parse, different
transport (OpenAI-compatible endpoints instead of local Ollama). Since then,
glyph's eval path has drifted further from gnomon-eval than G4 intended:

- The judge prompt/parse is **duplicated**, not imported — glyph carries its
  own copy of the text that defines what `faithfulness` and
  `context_precision` mean.
- The harness (`glyph/eval/harness.py:59-64`) implements its **own** numpy
  percentile bootstrap (2000 resamples) instead of gnomon's
  `aggregate_metric`, duplicating the statistical-honesty layer.
- `run_eval()` is bypassed entirely because gnomon's `EvalReport` does not
  expose per-case scores, which glyph needs for honest reporting (P3.5).

So today gnomon-eval is used as a *specification* (via
`GNOMON_GRAPHRAG_CONTRACT.md`), not as a dependency. Both duplicated pieces
can silently drift: a change in gnomon's prompt or CI math would stop glyph's
scores from meaning the same thing, with no test to catch it. Gnomon's
`aggregate_metric` was recently hardened (gnomon #36 property tests found and
fixed a real CI-ordering bug); glyph's private copy received none of that.

The open question this ADR settles: does glyph's own judge replace
gnomon-eval for glyph's evaluation, or does gnomon remain glyph's eval
engine?

## Decision

**GNOMON remains glyph's evaluation engine at the semantics layer. Glyph owns
transport only.** Concretely:

1. **Judge prompt/parse: gnomon's is canonical.** `OpenAICompatJudge` stays
   (transport is glyph's concern), but its prompt/parse must be gnomon's,
   imported or pinned by a contract test that fails when the two diverge.
   Glyph never edits metric semantics locally.
2. **Aggregation: use gnomon's `aggregate_metric`.** Replace the private
   numpy bootstrap in `harness.py` with gnomon's implementation — one
   statistical-honesty layer, the one with property tests and the CI-ordering
   fix. Glyph adds gnomon-eval as a dev/eval dependency (local path or git;
   offline-first is preserved, both repos are local).
3. **`run_eval()`: adopt when the gap closes.** Glyph keeps its custom
   per-case loop *only* because `EvalReport` hides per-case scores. File a
   gnomon issue to expose them (fits gnomon's roadmap Backlog B); when it
   lands, glyph's harness collapses to `run_eval` + the existing RagTarget
   adapter, and the custom loop is deleted.
4. **Metric ownership follows a two-layer boundary.** If a metric is
   computable from `(question, answer, contexts, expected_*)` — the
   RagTarget/RagResponse surface — it is a judge metric and belongs to
   gnomon (canonical; new needs proposed upstream, v2 backlog A1/A2
   territory). If it requires target internals the contract cannot see
   (hop counts, retrieval budgets, arm identity — e.g. hop-depth from the
   2026-07-13 spec, recall@budget from ADR-G4), it is glyph-native research
   instrumentation: out of gnomon's scope *by gnomon's own design*
   (ADR-0001, adapter-based black-box target), complementary to the judge
   layer, never a substitute for it. Glyph implements these freely without
   reopening this decision.

## Market context (build-vs-buy, scanned 2026-07-20)

Both projects' custom layers were checked against the popular stack (Ragas,
DeepEval, TruLens, promptfoo, MLflow evals; MS GraphRAG, LightRAG):

- **What justifies gnomon** (confirmed by two independent scans): none of
  the five eval frameworks ships native uncertainty quantification for
  LLM-as-judge metrics — best partials are MLflow's variance/p90
  aggregations and DeepEval/promptfoo reruns; the market pattern is
  "framework + your own stats layer on top". Gnomon's statistical-honesty
  invariant (no judge metric without a CI) is the differentiator. **What no
  longer does:** local/offline judges are commodity now (DeepEval
  `set-ollama`, promptfoo local-first) — offline-first is table stakes, not
  a moat. MLflow is the only one with real prompt/metric versioning
  (Prompt Registry, scorer versions) — relevant to trigger 1 below.
- **What justifies glyph** (claim narrowed by the deeper scan): honest
  field-wide graph-vs-vector benchmarks now EXIST — GraphRAG-Bench (strong
  rerank vector baselines, published config asymmetries, negative findings
  for MS-GraphRAG/LightRAG) and Han et al. 2026 (unified protocol,
  judge position-bias audit). Glyph's justification is therefore NOT "no
  one benchmarks honestly" but the narrower, sturdier one: a
  **corpus-specific decision instrument** — does graph structure pay its
  cost on the ecosystem's own corpora (AXON docs, RPG), with gnomon CIs
  and glyph cost gates — a question field-wide benchmarks don't answer.
  Research to-dos inherited: (a) keep strengthening the vector baseline
  (hybrid/reranker — GraphRAG-Bench's bar), (b) align/validate glyph's
  protocol against GraphRAG-Bench's disclosure practices, (c) consider a
  judge position-bias audit (Han et al.) if pairwise judging is ever
  introduced.
- **Commodity in both:** metric implementations themselves (faithfulness /
  context_precision exist everywhere). Neither project should carry
  duplicated commodity code — which is what this ADR eliminates.

## Decision matrix

Options: (A) status quo, gnomon as spec only; (B) this ADR; (C) full
`run_eval` adoption today; (D) full replacement, glyph owns everything.
Weights reflect what evaluation exists for. Scores 1-5.

| Criterion (weight) | A | B | C | D |
|---|---|---|---|---|
| Semantic integrity / comparable scores (5) | 2 | 5 | 5 | 1 |
| Inherited statistical honesty (4) | 2 | 5 | 5 | 2 |
| Maintenance cost / duplication (4) | 2 | 4 | 5 | 1 |
| Low coupling (3) | 5 | 3 | 2 | 5 |
| Velocity for new metrics (3) | 4 | 3 | 2 | 5 |
| Simplicity / less glyph code (3) | 3 | 4 | 5 | 2 |
| Reversibility (2) | 5 | 4 | 3 | 2 |
| **Weighted total (/120)** | **72** | **99** | **98** | **57** |

C is ineligible today (run_eval hides per-case scores) and is B's declared
end-state — approving B approves C with a viable ramp. B stays ahead of A
even zeroing its weakest criterion (new-metric velocity). D's genuine
advantage (research autonomy) is captured by the two-layer boundary in
decision 4 without paying D's price (judge-semantics fork).

## Revisit triggers

Reopen this decision if any of these becomes true:

1. **Market closes the gap:** Ragas/DeepEval ship native bootstrap CIs and
   versioned judge prompts → gnomon becomes a thin wrapper or retires;
   glyph follows whatever replaces it.
2. **Gnomon goes unmaintained:** revert to A (the dependency is dev-only;
   reversal is mechanical).
3. **Layer-1 divergence:** two judge-layer metric proposals rejected
   upstream in gnomon → reopen D for the judge layer only. (Layer-2
   instrumentation never triggers this — it is glyph-native by design.)
4. **External benchmark coverage:** if GraphRAG-Bench (or a successor)
   grows to cover the ecosystem's corpora and cost model well enough that
   running it locally answers glyph's decision question, shrink glyph's
   benchmark arm to an adapter over it instead of maintaining a parallel
   protocol.

## Alternatives considered

- **Full replacement (drop gnomon):** glyph owns judge + aggregation +
  metrics. Rejected: duplicates battle-tested statistical machinery, scores
  stop being comparable across the ecosystem (AXON is evaluated by gnomon),
  and every honesty fix must be made twice.
- **Status quo (gnomon as spec only):** zero coupling, but silent semantic
  drift is unguarded — the exact failure mode is already visible (glyph's
  bootstrap missed gnomon #36's hardening). Rejected.

## Consequences

- **Positive:** one source of truth for metric semantics and CI math; glyph
  inherits gnomon's hardening for free; ecosystem scores (glyph, AXON)
  remain comparable; glyph's eval code shrinks (custom bootstrap deleted now,
  custom loop deleted after the gnomon issue lands).
- **Trade-offs:** glyph gains a dependency on a local sibling repo (path/git
  pin — acceptable for a dev/eval extra, never a runtime dependency); glyph's
  eval cadence can be briefly blocked on gnomon changes (mitigated by the
  small, versioned contract surface).
- **To do on acceptance:** (a) contract test or import pinning the judge
  prompt; (b) swap `harness.py` bootstrap for `aggregate_metric`; (c) open
  the gnomon issue "expose per-case scores in EvalReport"; (d) mark ADR-G4's
  judge section as extended by this ADR.
