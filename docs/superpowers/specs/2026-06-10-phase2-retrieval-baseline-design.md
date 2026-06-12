# Design — Phase 2: Graph-aware retrieval + vector baseline (P2.1–P2.4)

**Date:** 2026-06-10
**Status:** Approved (brainstorming)
**Scope of this session:** P2.1–P2.4 (three retrieval arms + contract). **Measurement** is Phase 3
(depends on unblocking GNOMON in Phase 2.5) and is out of scope for this session.

## Objective

Build GLYPH's retrieval machinery: three arms — graph-aware, fair vector baseline, and hybrid —
producing a single output contract (`ContextPack`) comparable token-by-token across the same corpus
(the Monster Manual graph and the chunks that generated it). No benchmark in this phase; just the
modes working. Deliverable: `retrieve(query, mode=graph|vector|hybrid) -> ContextPack`.

## Closed decisions (brainstorming)

1. **Embeddings:** local multilingual sentence-transformers + in-memory vector index (numpy/cosine).
   Zero server, zero API cost, supports PT-BR. Optional extra `glyph-kg[retrieval]`. Default model
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (configurable).
2. **Fair corpus:** the vector baseline indexes the text of the **same ~425 creature chunks**
   (output of `chunk.by_creature` + `is_creature`). The graph arm is structured extraction from those
   same chunks. Same source, different representations — controlled experiment.
3. **Graph anchoring:** lightweight embedding — embeds the query and node labels, anchors to top-k
   nearest labels. Covers entity-naming queries ("what does Deva resist?") and conceptual queries
   ("which creatures resist fire?") because neighborhood expansion is undirected.
4. **Token budget:** all three arms truncate output at the same budget, measured by **character-based
   estimation** (declared limitation; real tokenizer is Phase 3 scope, aligned to AXON's honesty).
5. **Contract:** `Segment`/`ContextPack` in `glyph/model/`, identical across all three arms.

## Architecture (hexagonal, preserves Phase 0 invariants)

```
glyph/
  embed/          # embedding infrastructure — does not import model/store/retrieval/baseline
    port.py         # Embedder Protocol; VectorIndex Protocol
    sentence_transformers_embedder.py   # local adapter (optional extra)
    memory_index.py # InMemoryVectorIndex (numpy cosine)
  model/
    contract.py     # Segment, ContextPack (+ estimate_tokens helper)
  retrieval/        # depends on store(port) + embed + model; does NOT import baseline
    port.py         # Retriever Protocol: retrieve(query, ...) -> ContextPack
    graph.py        # GraphRetriever (anchor -> subgraph(hops) -> ContextPack)
    hybrid.py       # HybridRetriever(graph, vector) — fusion via injection, under Retriever Protocol
  baseline/         # depends on embed + model; does NOT import retrieval
    vector.py       # VectorBaseline (indexes chunks -> top-k cosine -> ContextPack)
```

Dependency rule: `embed` does not know anyone else; `retrieval` and `baseline` depend on `embed`,
`model`, and (retrieval) on the `GraphStore` port. **`retrieval` and `baseline` don't care**:
`HybridRetriever` receives two objects that satisfy the `Retriever` Protocol (injected via
composition root), then fuses without coupling the layers. Architecture invariant tests are
extended to cover `embed`/`retrieval`/`baseline`.

## Output contract

```python
class Segment(BaseModel, frozen=True):
    text: str
    source: str          # node id (graph) or chunk label (vector)
    score: float

class ContextPack(BaseModel, frozen=True):
    mode: Literal["graph", "vector", "hybrid"]
    segments: list[Segment]
    token_estimate: int

def estimate_tokens(text: str) -> int   # character-based estimation (declared)
```

The `ContextPack` is assembled by adding segments (ordered by score) until the token budget is reached;
`token_estimate` reflects the total included. Same budget across all three arms = fair comparison.

## Data flow per arm

- **GraphRetriever(store, embedder, node_labels_index):** embeds the query → top-k anchors (cosine vs
  labels) → `store.subgraph(anchor_ids, hops)` → for each node in the subgraph assembles a `Segment`
  (label + attributes + its edges, e.g., "Deva — resists radiant; immune_to enchanted") →
  sorts by anchor proximity → truncates to budget.
- **VectorBaseline(embedder):** `index(chunks)` embeds each chunk text into `InMemoryVectorIndex`
  → `retrieve(query, budget)` embeds the query, searches top-k, assembles `Segment` per chunk → truncates to budget.
- **HybridRetriever(graph, vector):** runs both, fuses via **reciprocal rank fusion** on
  `Segment.source`, deduplicates, truncates to shared budget.

## ADR-G3 (to be recorded before implementation)

Fair baseline: same chunks, same embedder, same budget, real implementation (chunk + embedding
+ vector store + top-k), not a strawman. Documents budget equality across arms and the
character-based token counting limitation in this phase.

## Test strategy (TDD)

- All logic — anchoring, `ContextPack` assembly, budget truncation, RRF fusion, cosine search
  in `InMemoryVectorIndex` — tested with a **deterministic fake embedder** (fixed vectors per
  text). No model downloads, no network.
- `InMemoryVectorIndex`: cosine/top-k tests with known vectors.
- The real `SentenceTransformerEmbedder` adapter gets an **opt-in** smoke test (`@pytest.mark.slow`,
  outside default CI — downloads the model) that embeds two PT-BR sentences and verifies the most
  similar ranks first. Mirrors the fake-LLM/`@pytest.mark.live` pattern from Phase 1.
- Manual validation (outside CI): a wiring script loads `out/monster-manual.json` into a
  `NetworkXStore` + MM chunks, and runs a few queries in all three modes for inspection.

## Phase 2 done criteria

- ADR-G3 committed before implementation.
- `embed` (port + ST adapter + InMemoryVectorIndex), `Segment`/`ContextPack`, `GraphRetriever`,
  `VectorBaseline`, `HybridRetriever` implemented with TDD; test suite green; architecture invariants
  extended and green.
- All three modes run against real MM graph/chunks via wiring script (manual inspection).
- CI green (lint, types, test, coverage, invariants). Tests that download models stay out of default.

## Out of scope

Benchmark/measurement (Phase 3), GNOMON (Phase 2.5), real tokenizer, AXON integration, complete `Glyph`
facade from README (the three retrievers + wiring script suffice; the facade enters when it adds value).
