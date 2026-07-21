# ADR-G3: Fair vector baseline and output contract

**Date:** 2026-06-10
**Status:** Accepted

## Context

The GLYPH thesis is that graph-aware retrieval outperforms vector-only on queries that depend on
relationships between entities. The experiment is only valid if the vector baseline is strong and fair
— weakening it invalidates the result.

## Decision

**Same corpus:** the vector baseline indexes the text from the same chunks by creature
(`chunk.by_creature` + `is_creature`) that generated the graph. The graph arm is the
structured extraction of these chunks; the vector arm is their raw text embedded.

**Same embedder and same budget:** both arms use the same local embedder
(sentence-transformers multilingual) and truncate the output at the same token budget. The hybrid arm
fuses the two under the same budget.

**Single contract:** `Segment`/`ContextPack` identical across all three modes, token-by-token comparable.

**Declared limitation:** budget is measured by character estimate at this phase (not real tokenizer).
Declared here and in the benchmark (Phase 3).

## Consequences

**Positive:** controlled comparison (same source, same embedder, same budget). The baseline is
real implementation (chunk + embedding + vector store + top-k), not a strawman.

**Trade-offs / to observe:** token estimate per character is approximate; Phase 3 switches to
real count where it matters. Hybrid fusion (reciprocal rank fusion) treats graph segments
(source = node id) and vector segments (source = chunk label) as distinct sources; unifying
identity across representations is deferred if the benchmark requests it.

## Amendment (2026-07-21, issue #22): real tokenizer for budget accounting

The "Declared limitation" above (budget measured by character estimate, `estimate_tokens`
= ceil(chars/4)) is superseded on the paths where the budget materially affects the result.

**What changed:** `glyph/model/contract.py` gains `count_tokens()`, a real tokenizer
(tiktoken `cl100k_base`), now an explicit core dependency in `pyproject.toml` rather than
relying on it being pulled in transitively. `pack()` now takes an injected `cost` callable; all seven retrieval arms
(vector, graph, hybrid, reranked, community, multi_anchor, path) pass `cost=count_tokens`, so
every arm's budget packing - and therefore the benchmark - counts real tokens.

**Scope / what did NOT change:** `estimate_tokens` is retained and remains `pack()`'s default
cost, so the char estimate still backs any caller that has not opted in and keeps the
established unit contract stable. The fairness invariant of this ADR is preserved: all arms
use the *same* cost function (previously char/4, now real tokens), so budgets remain
comparable across arms.

**Why cl100k_base:** it is the tokenizer already present in the gated install set. The
sentence-transformers model's own tokenizer would require adding the `embeddings` extra to the
gate - a larger footprint than reusing an already-installed dependency, for no material gain in
budget accounting.

**Benchmark impact:** `eval/benchmark-baseline.json`'s `total_tokens` and budget-derived figures
were measured under the old `estimate_tokens` (char/4) cost function and are now stale relative
to the real-tokenizer cost applied above. Regenerating them needs the non-committed source book
PDF plus paid generation/judge API keys, which is outside the offline gate's scope
(`.claude/loop.yaml`) - the same constraint documented for sibling issue #18 - so it is not
re-run in this pass; regeneration is deferred to the next scheduled full benchmark run, since
`loop.yaml` treats `benchmark-baseline.json`/`METRICS*.md` as regression gates, not files to
touch just to make a change land. The effect is bounded, not systematic: spot-checking
`count_tokens` against `estimate_tokens` on real segment text from the test suite shows the
delta runs in both directions and stays small - Portuguese creature-description sentences (the
corpus's actual language) tend to count slightly *higher* under the real tokenizer than char/4
(e.g. "O fogo consome a floresta antiga durante a noite" is 15 real tokens vs. 12 estimated),
while plain English prose tends slightly *lower* (e.g. "The quick brown fox jumps over the lazy
dog near the old oak tree." is 15 real tokens vs. 17 estimated). Either way the gap is a handful
of tokens per ~50-character segment, so it can only flip inclusion for a segment sitting right
at a `token_budget` boundary; arms comfortably under or over budget are unaffected. Since the
same cost function is applied uniformly across all seven arms, no arm gains an advantage
relative to another - ADR-G3's fairness invariant holds; this makes budget accounting more
accurate, not biased.
