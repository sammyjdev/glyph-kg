# GLYPH Architecture

## Layered View

GLYPH follows hexagonal architecture: the domain (graph) at the center, extraction and persistence as ports with interchangeable adapters.

```
glyph/
  model/        # domain: Node, Edge, EdgeType, NodeType (Pydantic v2)
  extract/      # Extractor port + adapters (document, code)
  store/        # GraphStore port + adapters (networkx, neo4j)
  retrieval/    # graph-aware retrieval over the port
  baseline/     # vector baseline (benchmark control)
  eval/         # GNOMON integration, metrics, CIs
```

Dependency rule: arrows point inward. `extract`, `store`, `retrieval`, `baseline`, and `eval` depend on `model`. `model` depends on nothing. No adapter knows another adapter.

## Graph Core (model)

`Node`, `Edge`, `EdgeType`, `NodeType` in Pydantic v2.

Types separated by domain to avoid creating a generic schema that serves none:
- Code: `NodeType` in {File, Module, Class, Function}; `EdgeType` in {DEFINES, IMPORTS, CALLS, INHERITS, REFERENCES}.
- Document: `NodeType` in {Entity, Concept, Section}; `EdgeType` in {RELATES_TO, MENTIONS, REQUIRES, RESISTS, ...} as appropriate for corpus domain.

The graph is origin-agnostic: once built, retrieval and store treat nodes and edges equally, regardless of which extractor produced them.

## Extractor port

```python
class Extractor(Protocol):
    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]: ...
```

Two adapters:

- **DocumentExtractor** (probabilistic). PDF ingestion (pymupdf or similar) → structure-aware chunking → entity/relation extraction prompt to an LLM → parse to `Node`/`Edge`. Extraction has error; this is why the benchmark measures quality instead of assuming it.
- **CodeExtractor** (deterministic). tree-sitter for Python and Java, aligned with `graph_extractor.py` from AXON. The relation `A CALLS B` is a fact extracted from the AST, not inference. Limitation: symbol resolution by name in import graph + intra-file, without full cross-language type inference.

The probabilistic/deterministic asymmetry is why there are two adapters instead of a generic extractor.

## GraphStore port

```python
class GraphStore(Protocol):
    def upsert_nodes(self, nodes: Sequence[Node]) -> None: ...
    def upsert_edges(self, edges: Sequence[Edge]) -> None: ...
    def neighbors(self, node: NodeId, hops: int) -> Subgraph: ...
    def subgraph(self, seed: Sequence[NodeId], hops: int) -> Subgraph: ...
    def shortest_path(self, src: NodeId, dst: NodeId) -> Path | None: ...
```

- **NetworkX** (default): in-memory with persistence (graphml/pickle). Zero server, pip-installable. The graph of a document corpus fits in memory; for code, also at the scale of target repos.
- **Neo4j** (adapter): **implemented and smoke-tested** (14/14 contract tests vs Neo4j 5 on Testcontainers), openCypher. Exists for CV keyword and production history, not as default. Not an always-on service of the project.

Both pass the same set of contract tests. Swapping the backend does not change results, only performance/scale.

## Graph-aware Retrieval

Given a query: anchor relevant entities in the graph (by node match or light embedding of the label), expand the neighborhood by `hops`, construct context from the subgraph. The returned context is structural: the entities connected to the anchor, not the most similar chunks.

Output in unified contract (`Segment`/`ContextPack`) comparable token-for-token with the baseline.

## Vector Baseline (Control)

Fair and real implementation in Python over the **same** corpus: chunk + embedding + vector store + top-k by similarity. Same token budget as the other arms. This is the experiment control; weakening it invalidates the benchmark. There is a third hybrid arm (graph + vector fusion).

## Eval (GNOMON)

Measures the three arms (graph, vector, hybrid) over a query set from the corpus. Metrics: `faithfulness` and `context_precision` (from GNOMON v1), token efficiency, cost, and latency. All with confidence intervals via percentile bootstrap. Reproducible result from a versioned fixture.

GNOMON is pull-based: `run_eval` calls `target.query(question)`. GLYPH does not invert this — it pre-computes the results of each arm and exposes them behind an adapter **`RagTarget`** (one per arm) that satisfies GNOMON's Protocol and returns the stored result. `RagResponse` requires `total_tokens` and `latency_ms`, so the three arms instrument real token and latency; cost in US$ is derived from tokens in GLYPH. The adapter lives in GLYPH (`eval/`), not in GNOMON — no change to GNOMON is necessary for the benchmark.

## AXON Integration

AXON consumes GLYPH as a dependency. AXON's `GraphContextSource` (ADR-102 in AXON's repo) delegates to `glyph.retrieval`. AXON's consolidated code graph (ADR-103) is one of the sources that `CodeExtractor` feeds or consumes. This maintains a single canonical source for graph logic.

## Invariants Verified in CI

1. `model` does not import anything from `extract`, `store`, `retrieval`, `baseline`, `eval`.
2. No extract adapter imports store adapter, and vice versa.
3. `retrieval` depends on the `GraphStore` port, not a concrete adapter.
4. The vector baseline and graph retrieval share the same output contract and the same token budget.
