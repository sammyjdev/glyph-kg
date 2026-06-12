# ADR-G6: GraphContextSource is the product boundary of GLYPH

**Date:** 2026-06-11
**Status:** Accepted

## Context

GLYPH is now used as a **product library** by external consumers (AXON via dec-116, and future
clients), not just by its own benchmark. A consumer needs a **named and stable** boundary for
graph-aware retrieval, rather than depending on the construction details of `GraphRetriever`
(embedder + node list + `hops`/`anchors`), which are internal and may evolve (hybrid fusion,
different anchoring, node-listing changes).

The `glyph.integration.GraphContextSource` already existed as a thin facade, but exposed the
`context()` method, which was parallel — and therefore **did not satisfy** the canonical port
`glyph.retrieval.port.Retriever` (`retrieve(query, token_budget) -> ContextPack`). Two names
for the same contract is precisely the incoherence that a product library should not have.

## Decision

- `GraphContextSource` **satisfies the `Retriever` port**: the external method is `retrieve(query,
  token_budget) -> ContextPack`, identical to the contract of all branches. Thus the facade is a
  structural `Retriever` and fits wherever a `Retriever` is expected. A test locks in the
  conformance (`isinstance(source, Retriever)`).
- **Two entry points, same object:**
  - `GraphContextSource(store, embedder, nodes)` — **in-memory**: the caller already has a
    `GraphStore` and node list (as in AXON, which builds them from a SQLite graph).
  - `GraphContextSource.from_graph_file(path, embedder)` — **persisted**: loads a NetworkX graph
    (file or code) from disk, folding load + node-listing + wiring into one call.
- The boundary can evolve behind `retrieve()` without breaking consumers. `nodes` remains explicit
  in the constructor (the `GraphStore` port does not enumerate nodes, and `GraphRetriever` already
  requires them); we do not invent a listing method on the port just to hide this argument.

## Consequences

- GLYPH exposes **one** retrieval contract (`Retriever`), without a parallel method.
- AXON (dec-116) now delegates to `GraphContextSource(...).retrieve(...)` instead of instantiating
  `GraphRetriever` directly — it depends on the boundary, not on internals (change tracked in the
  AXON repo).
- Renaming `context()` → `retrieve()` is an API break; since the only consumer (AXON) was not yet
  using the facade, there is no compatibility to maintain. Future boundary changes become
  conscious decisions recorded here.
