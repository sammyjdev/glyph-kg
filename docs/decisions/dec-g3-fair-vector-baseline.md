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
