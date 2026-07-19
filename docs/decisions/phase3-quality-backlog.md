# Extraction quality backlog — revisit in Phase 3

Quality items observed at the P1.4 gate (Monster Manual, see
`phase1-cost-gate-results.md`). Do not block Phase 2; enter Phase 3 evaluation,
where document extraction quality is measured (not assumed).

> **Ticketed 2026-07-18 as issues #18–#22** (casing, attack-vs-resistance eval,
> asymmetry declaration, attrs-in-segments experiment, real tokenizer).
> Left out: item 3 (observe-only, no current corpus triggers it) and the demo
> embedding cache (deferred until it becomes recurring use).

## 1. Node label casing normalization

Labels come from the LLM-returned name in inconsistent cases (`ANKHEG`, `abolete`,
`Deva`). The **ids are already normalized** (lowercase + collapsed spaces), so
deduplication works — the problem is purely cosmetic in the `label`.

**Suggestion:** normalize the `label` in display/persistence (e.g., title-case with exceptions
for articles/prepositions, or preserve the most frequent form seen). Evaluate whether it affects
measured relevance or only presentation.

## 2. Probabilistic relation errors

Extraction is probabilistic and has error — that is why the benchmark measures it. Gate example:

- **ANKHEG: `resists ácido`** — likely confusion between the creature's *acid* attack and
  an *acid* resistance.

**Suggestion (Phase 3):** include in the query set cases where the text distinguishes attack vs.
resistance/immunity, and report the error rate of this type as part of the honest number
(P3.5). Consider a stronger verification/few-shot step in the prompt if the rate matters.

## 3. (to observe) Section headers in creature source

The `is_creature` filter (attribute block) already removed rules/index sections. If a
future book (PHB/DMG) lacks an attribute block, this signal will not work — own schema and filter
will be necessary (P1.5/future phase).

## 4. Retrieval (Phase 2) — declare/observe in benchmark

From Phase 2 final review. The cheaper correction items have already been resolved in the
hardening commit (identity unification in hybrid via case-insensitive key; deterministic tie-break by `source` in graph). Remaining for Phase 3:

- **Asymmetric embedding surface:** the graph arm embeds only node labels (names); the
  vector arm embeds the entire chunk text (stats block). It is by design (the graph value
  lies in neighbor expansion, not anchor recall), but must be **declared in
  the paper** so the comparison is not read as "representation only".
- **Coverage per budget differs between arms:** graph segments (`label — relations`) are
  much shorter than vector chunks and ignore `Node.attrs` (CR/type/alignment extracted
  but unused). Under the same char budget, the graph packs more segments. Not unfair,
  but affects recall@budget interpretation — declare, and consider including attrs in the graph segment.
- **Real tokenizer:** budget is char estimation (ADR-G3). Switch to real count where it matters.
- **Demo without embedding cache:** `scripts/retrieve_demo.py` re-embeds all node labels at
  each execution; slow on the full graph with the real ST model. Cache if it becomes recurring use.
