# Spec P7 — Global community axis (GLYPH)

**Date:** 2026-06-11
**Status:** Implemented + **measured** (GLYPH side) — see dec-g7. Global benchmark run (n=8, two judges): community summary yields equal or better quality to ~½ of the tokens. AXON orchestration = sister spec pending.
**Associated ADR:** [dec-g7](dec-g7-global-community-axis.md)
**Boundary:** logic and detection in GLYPH (pure lib, LLM injected, mirrors `DocumentExtractor`). Orchestration (post-commit hook, MCP `get_global_context`, provider/API key) is the sister spec in AXON — **outside this spec**.

---

## 1. Objective

Add a **global axis** to retrieval: detect communities in the graph, summarize them via injected LLM, and retrieve those summaries for thematic/sense-making questions ("what are the major subsystems in this repo?") — which local neighborhood expansion (`GraphRetriever`, 2 hops) does not answer well.

Local answers "what depends on X?"; global answers "how is this organized?". They are complementary axes behind the same `Retriever` port.

## 2. Core decisions (locked)

1. **Logic in GLYPH, LLM injected.** The lib owns detection + summarization; receives an `llm` as a dependency (no provider/API key in the lib), mirroring `DocumentExtractor`. AXON injects the model and fires it in the hook.
2. **Communities live in the same graph** as `COMMUNITY` nodes + `CONTAINS` edges (community→member).
3. **Isolation by traversal projection, NOT by disjoint indexing.** ⬅️ *critical correction vs. original design — see §4.*

## 3. Components

### 3.1 Model (1 line each)

- `glyph/model/node.py` → `NodeType`: add `COMMUNITY = "community"`.
- `glyph/model/edge.py` → `EdgeType`: add `CONTAINS = "contains"`.
- `glyph/model/contract.py` → `Mode`: `Literal["graph", "vector", "hybrid", "community"]`.

### 3.2 `glyph/retrieval/community.py` (new)

```python
# Allowlist default = structural code subgraph. DO NOT include
# document/decision edges (MENTIONS, RELATES_TO, etc.) else clusters mix ADR with function.
STRUCTURAL_EDGES = frozenset({
    EdgeType.DEFINES, EdgeType.IMPORTS, EdgeType.CALLS,
    EdgeType.INHERITS, EdgeType.REFERENCES,
})

def detect_communities(
    store, nodes, *, seed: int,
    edge_types: frozenset[EdgeType] = STRUCTURAL_EDGES,
) -> list[Community]:
    """louvain_communities(seed=seed) over the structural subgraph.
    - EXCLUDES pre-existing COMMUNITY nodes (idempotence — see build §5).
    - Feeds the graph in DETERMINISTIC ORDER (sorted by node.id); seed
      alone does NOT reproduce without fixed iteration order in networkx.
    - Filters edges by `edge_types`.
    """

def to_graph_elements(communities) -> tuple[list[Node], list[Edge]]:
    """COMMUNITY nodes + CONTAINS edges (community→member), for upsert in the same store.

    community id = "community:" + hash(tuple(sorted(member_ids)))   ⬅️ NOT index N.
    attrs={"members": k}. (summary/title filled in summarize_communities.)

    Reason for hash: stable id across rebuilds → unchanged communities keep the id
    → summarize SKIPS the unchanged ones → cuts LLM cost in hook (leverage, not cosmetic).
    """

def summarize_communities(communities, member_text, llm) -> list[Node]:
    """Assembles prompt from members, calls the injected `llm`, returns COMMUNITY nodes with
    attrs["summary"] AND attrs["title"] (1-line thematic) filled in.
    Mirrors DocumentExtractor: no provider/API key here.

    title: cheap hedge against embedding dilution (multi-sentence summary in a single
    vector dilutes the centroid). v1 anchors on summary; title is ready for a future
    retriever to swap without schema migration.
    """
```

### 3.3 `CommunityRetriever(Retriever)` (in `community.py`)

```python
class CommunityRetriever:
    def __init__(self, community_nodes, embedder):
        # embeds attrs["summary"] of each COMMUNITY node (COMMUNITY nodes only).
    def retrieve(self, query, token_budget=1000) -> ContextPack:
        # anchors the query on the summaries and returns them as Segments.
        # mode="community". NO subgraph expansion (no _store.subgraph).
        return pack("community", segments, token_budget)
```

Satisfies the `glyph.retrieval.port.Retriever` port (`retrieve(query, token_budget) -> ContextPack`), like the other branches — conformance test `isinstance(r, Retriever)` as in dec-g6.

## 4. Isolation — the critical correction

**The hole in the original design:** "disjoint indexing" only isolates *anchoring*. But `GraphRetriever.retrieve` calls `store.subgraph(anchors, hops)`, and `NetworkXStore.subgraph` (`networkx_store.py:49`) does BFS over the **entire graph**:

```python
reachable.update(nx.single_source_shortest_path_length(undirected, src, cutoff=hops))
```

Thus, even anchoring only on non-COMMUNITY nodes, hop expansion traverses `CONTAINS` edges. Two effects:

1. **Leakage:** COMMUNITY node enters the local ContextPack.
2. **Distortion (worse):** each COMMUNITY becomes a **super-hub** linked to all members → any two symbols in the same community are 2 hops apart (member→COMMUNITY→member). Corrupts ALL distances in local retrieval.

**Fix (same graph, pruned projection) — decision B, implemented in full:**

- Generic params `exclude_node_types` / `exclude_edge_types` (keyword, default empty) on **all three** traversal methods of `GraphStore`: `subgraph`, `neighbors`, `shortest_path`. The store stays dumb — only honors a filter, **doesn't know** what COMMUNITY is (keeps the port swappable; the Neo4j adapter doesn't need to know about the overlay). The **policy lives in the retrieval layer**.
- The `GraphRetriever` passes `{COMMUNITY}` / `{CONTAINS}` (constants `_OVERLAY_*` in the module). The global layer (`CommunityRetriever`) **doesn't traverse** (direct anchoring on summaries), so doesn't need the projection.
- The MCP tools in AXON (`get_graph_neighbors`/`get_graph_path`) pass the same overlay — **in the AXON sister spec**, not here.
- Footgun (forgetting to exclude) covered by the **topological invariance test** (§6.5), not by coupling the store.

**Etch in dec-g7:** "isolation by **pruned traversal projection**, not disjoint indexing; generic filter on store, policy in retrieval layer (option B)."

## 5. Data flow

- **Build** (AXON fires after):
  `load → detect_communities(seed) → to_graph_elements → upsert → summarize_communities(llm) → upsert COMMUNITY with summary+title → persist`.
  **Idempotence:** before detecting, remove pre-existing `COMMUNITY` nodes + `CONTAINS` edges (or `detect` excludes COMMUNITY from input **and** build cleans the old ones), else accumulates on each commit.
- **Query:** `CommunityRetriever.retrieve(query_global) → summaries as ContextPack`.

## 6. Tests / validation (of this spec — 100% gate, zero cost)

Fake LLM injected + fake embedder:

1. **Detection** partitions a small known graph (assert expected partition, reproducible with fixed seed + deterministic order).
2. **Summarize** fills `attrs["summary"]` and `attrs["title"]` via stub.
3. **CommunityRetriever** anchors to the right community (thematic query → correct summary on top).
4. **Isolation — leakage:** local retrieval does NOT contain COMMUNITY nodes.
5. **Isolation — topological invariance** ⬅️ *new, mandatory*: on **realistic topology** (communities of size >2, structurally distant nodes), two distant nodes that fall in the same community **remain distant** in the local subgraph after build. Catches the super-hub effect that test 4 doesn't. (Topology doesn't need real LLM — just stub.)
6. **Port conformance:** `isinstance(CommunityRetriever(...), Retriever)`.

## 7. Follow-ups (outside this spec)

- **Real** summarization round (corpus Monster Manual / D&D — costs LLM).
- **Global benchmark** branch (depends on a global query set that doesn't yet exist).
- Eventual anchoring swap `summary` → `title`/title-weighted **if** benchmark shows dilution.

## 8. AXON boundary (sister spec, not here)

- Post-commit hook fires `detect + summarize + persist` with AXON's LLM.
- New MCP tool `get_global_context` (separate from `get_graph_context`).
- Caller chooses local vs global — **without automatic router in v1**.

## 9. dec-g7 (ADR to write together)

Records: global axis; native Louvain + seed + deterministic order; **in-graph isolation by pruned traversal projection** (not disjoint indexing); community id by hash of members (re-summarization leverage); summarization boundary (GLYPH logic / injected LLM); `title`+`summary` on COMMUNITY node.
