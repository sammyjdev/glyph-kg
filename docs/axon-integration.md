# AXON ↔ GLYPH integration (Phase 5)

GLYPH is the canonical knowledge-graph source; AXON consumes it instead of
re-implementing graph retrieval. This page records the **GLYPH side** of the contract.
The matching changes inside the AXON repo (ADR-102/103) are tracked there, not here.

## The seam GLYPH provides

`glyph.integration.GraphContextSource` is the single entry point AXON delegates to. It
satisfies the `Retriever` port (`retrieve(query, token_budget) -> ContextPack`), so it
drops in anywhere a retriever is expected.

```python
from glyph.integration import GraphContextSource
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder

# Persisted graph (document or code) from disk:
source = GraphContextSource.from_graph_file("out/monster-manual.json", SentenceTransformerEmbedder())

# Or in-memory, when the caller already holds a GraphStore + node list (AXON's case):
# source = GraphContextSource(store, embedder, nodes)

pack = source.retrieve("how does spell resistance interact with elemental damage", token_budget=1000)
# pack.segments -> graph-aware context, the same ContextPack contract as every benchmark arm
```

- `retrieve(query, token_budget)` returns a `ContextPack` — the same unified output the
  vector and hybrid arms produce, so AXON's downstream is contract-stable.
- `GraphContextSource(store, embedder, nodes)` is the in-memory entry; `from_graph_file`
  loads a persisted NetworkX graph (document **or** code — both the same core), folding the
  load + node-listing + retriever wiring into one call.

## What stays on the AXON side (out of this repo's scope)

- **P5.1 / P5.2:** AXON's own `GraphContextSource` (ADR-102) delegates to the class above,
  and the ADR gets a one-line note pointing here. These edits live in the AXON repo.
- **P5.3:** AXON's graph consolidation (ADR-103) stays; GLYPH consumes the consolidated
  graph through `from_graph_file` (point it at AXON's exported graph).
- **P5.4:** the end-to-end test (AXON serves graph-aware context via GLYPH over MCP) is an
  AXON integration test — it depends on AXON's MCP server, which this repo does not contain.

## Why this shape

The facade is deliberately thin: it adds a **stable, named boundary** so AXON depends on
`GraphContextSource.retrieve()` rather than on `GraphRetriever`'s constructor details (which
need an embedder and a node list wired up). The boundary can evolve (hops, anchors, hybrid
fusion) without breaking AXON.
