# Phase 3 experiment: include Node.attrs in graph segments (#21)

**Date:** 2026-07-21
**Status:** Decided — REVERT (attrs not included in graph segments)

## Context

`docs/decisions/phase3-quality-backlog.md` item 4's second bullet flagged that
graph segments (`label — relations`) ignore `Node.attrs` (CR/type/alignment,
extracted but unused) and suggested measuring whether including them changes
`recall@budget`. This doc records that experiment (issue #21).

`recall@budget` is deterministic, judge-free retrieval instrumentation
(`glyph/eval/recall.py`) — glyph-native research instrumentation explicitly
out of gnomon's judge scope per `dec-g8-gnomon-as-eval-engine.md` (it needs
target internals — retrieval budgets, segment provenance — the judge contract
cannot see), and per `dec-g4-eval-methodology.md`, context recall is "v2, not
built" in gnomon's judge metrics. No LLM/judge/generation call is required or
was made to produce this measurement.

## Architectural fact (why the outcome was expected)

`GraphRetriever` embeds and scores **node labels only**
(`self._label[node_id]` built in `__init__`, searched in `retrieve()`).
Segment **text** is never re-embedded or re-scored — it only feeds
`pack()`'s greedy, budget-bounded fill (`glyph/model/contract.py`). Appending
attrs to segment text therefore has exactly one causal effect: it lengthens
`Segment.text`, raising its token cost in the packer. That can only push
segments **out** of the budget — it cannot add a new relevant node or improve
a score. `recall@budget` can only stay flat or decrease when attrs are added;
it is architecturally impossible for it to improve under this design.

This does **not** mean attrs are worthless. It means they cannot help *this*
metric. Whether attrs improve answer quality (faithfulness / context_precision,
which require a generation + judge step) is unmeasured and out of scope for
this issue.

## Measurement

Script: `scripts/attrs_recall_experiment.py` (real `SentenceTransformerEmbedder`,
real `NetworkXStore.load("out/monster-manual.json")`, real `count_tokens`
via tiktoken — no LLM/judge calls). Two `GraphRetriever` arms over the same
store/embedder/nodes: `include_attrs=False` (baseline) vs `include_attrs=True`.

- **Subset:** the `graph_favored: true` queries from the frozen 25-query set
  (`eval/queries.json`) — **n = 18**.
- **token_budget:** 1000 (same for both arms).
- **Result (real run, 2026-07-21):**

  | arm            | mean recall@budget |
  |----------------|--------------------|
  | baseline       | 0.3485             |
  | attrs-included | 0.3265             |
  | **delta**      | **-0.0220**        |

## Decision

Using the proposed default threshold `delta >= 0.02 → KEEP, else REVERT`
(a default, not silently-invented policy — a human can override it): the
measured delta is **-0.0220**, below the threshold.

**Decision: REVERT.** Attrs are not included in graph segment text.
`GraphRetriever.include_attrs` stays `False` by default and unused in
production call sites (`scripts/run_benchmark.py` never passes it). Per the
`# ponytail:` comment on the flag in `glyph/retrieval/graph.py`, if this
experiment is ever discarded entirely the flag and the attrs branch can be
deleted; for now it is kept (behind the default-off flag) so the measurement
is reproducible without re-implementing it.

## Cross-references

- `docs/decisions/phase3-quality-backlog.md` item 4, second bullet — where
  this experiment was proposed.
- `docs/decisions/dec-g8-gnomon-as-eval-engine.md` — why recall@budget is
  judge-free glyph-native instrumentation, not a gnomon metric.
- `docs/decisions/dec-g4-eval-methodology.md` — context recall is v2 in
  gnomon's judge, not built; this experiment does not create or promise that
  metric, it only measures a code-level property of the packer.
