# SPEC: GLYPH

> Unified knowledge graph library for documents and code. Common graph core, two pluggable extractors under a port, graph-aware retrieval, measured against fair vector baseline with confidence intervals. Document-first.

## Problem

Vector retrieval recovers by semantic similarity and ignores explicit relations between entities. For relation-dense corpora (technical documents, rules) and for code (calls, imports, inheritance), structural context matters: the right answer is often one or two edges away, not in the most similar chunk. GLYPH builds the graph of these two domains under a single abstraction and measures the gain of graph-aware retrieval over vector-only.

## Measurable thesis

Graph-aware context outperforms vector-only in relevance and/or token efficiency for queries that depend on relations between entities. Where the query is simple factual, the vector may suffice. GLYPH reports both cases, with cost and latency as first-class metrics.

## Design principle

The two domains share graph core, store, retrieval, and measurement. They differ only in **extraction**: document is probabilistic (LLM reads prose, infers relations with error), code is deterministic (tree-sitter, the relation is fact). The right boundary is an `Extractor` port with two adapters, not a single schema trying to serve both.

## In Scope

1. **Graph core**: `Node`/`Edge`/`EdgeType` model (Pydantic v2), generic enough for both domains, with edge types separated by domain.
2. **`GraphStore` port**: NetworkX as default embedded (pip-installable, zero server), Neo4j as smoke-tested adapter.
3. **`Extractor` port** with two adapters:
   - `DocumentExtractor` (LLM): PDF ingestion, structure-aware chunking, entity/relation extraction.
   - `CodeExtractor` (tree-sitter): Python + Java, aligned to AXON's `graph_extractor`.
4. **Graph-aware retrieval**: entity anchoring + neighborhood expansion by `hops`.
5. **Fair vector baseline** (Python, same corpus): chunk + embedding + vector store. Real experiment control.
6. **GNOMON benchmark**: graph vs vector vs hybrid, with CIs (percentile bootstrap), cost and latency. GNOMON (audited) is already pip-installable and pull-based (`run_eval` calls `target.query`); GLYPH consumes it via a **`RagTarget`** adapter per arm, which returns precomputed results. v1 contract restrictions, declared: `RagResponse` requires `total_tokens` and `latency_ms` (token/latency instrumented across three arms, requirement); v1 metrics = `faithfulness` + `context_precision` only (no recall); cost in US$ calculated by GLYPH from tokens.
7. **Reproducible baseline**: versioned dataset/fixture + regeneration + regression check on published number.

## Out of Scope (honesty in claims)

- Complete cross-language type inference in code. Resolution by name in import graph + intra-file. Declared.
- Real tokenizer where we inherit char-estimate counting from AXON. Declared in benchmark.
- Dedicated external graph backend beyond the Neo4j adapter. Default stays embedded (local-first).
- Reimplement parsing that AXON already has. The `CodeExtractor` aligns instead of duplicating.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). Summary: Extractor port (2 adapters) → graph core (Node/Edge) → GraphStore port (NetworkX/Neo4j) → graph-aware retrieval → GNOMON benchmark against vector baseline.

Invariants (verified in CI):
- Extractors implement the port; the graph core does not import concrete extractor.
- Retrieval and store depend on the domain model, not the extractor.
- The vector baseline is fair implementation, not a straw man.

## Validation

| Layer | Criterion |
|---|---|
| Technical | Ports work; NetworkX and Neo4j pass the same smoke test; retrieval deterministic and reproducible |
| Evidence | Relevance, token efficiency, cost, latency: graph vs vector vs hybrid, with CIs, on real corpus |
| Honesty | Reproducible baseline from repo; limitations declared; benchmark not asserted before running |

Document validation corpus: 10-15 D&D books (150-300 pages), real data and dense in entities. Cost gate on 1 book before scaling (LLM extraction is expensive at volume).

## Phasing

Details in [GLYPH_PLAN.md](GLYPH_PLAN.md). Summary: Phase 0 foundation → Phase 1 document extraction → Phase 2 retrieval + baseline → Phase 3 benchmark (claim of completeness) → Phase 4 code → Phase 5 AXON integration → Phase 6 publication. (The old Phase 2.5 "unblock GNOMON" was dissolved by audit — GNOMON is already pip-installable; it becomes sub-task P3.0.) Each phase has an isolated publishable deliverable.

## Quality gates

- TDD per sub-task.
- Coverage with gate in CI.
- Architecture invariant tested.
- ADR per architectural decision (ADR-G1..G5).
- Cost measured before scaling extraction.
- Published number reproducible from repo, otherwise not published.

## Open decisions (do not block Phase 0)

- Embedding lib and vector store of baseline (Phase 2): decide between sentence-transformers + pgvector or what GNOMON/AXON already use, for consistency.
- ~~GNOMON packaged (Phase 2.5): package vs vendor~~ — **resolved by audit:** GNOMON is already pip-installable; GLYPH only references it by path/git and consumes via `RagTarget` adapter. Phase 2.5 dissolved (becomes P3.0).
- License (MIT suggested for public portfolio).
