# ADR-G7: Global community axis

**Date:** 2026-06-11
**Status:** Accepted

## Context

Local retrieval (`GraphRetriever`) anchors the query and expands the neighborhood by `hops` —
excellent for *"what depends on X?"*, poor for thematic/sense-making questions
(*"what are the major subsystems?"*). Missing a **global axis**: detect communities in the
graph, summarize them, and retrieve those summaries. Mirrors GraphRAG global, but within
GLYPH's profile (pure lib, reproducible, validation-first, same `ContextPack` contract).

## Decision

- **Detection: Native Louvain from networkx** (`nx.community.louvain_communities`) — zero
  new dependency (vs. Leiden, which would bring `igraph`/`leidenalg`). Operates on the **structural
  projection** of the graph (`STRUCTURAL_EDGES`: DEFINES/IMPORTS/CALLS/INHERITS/REFERENCES) —
  document/decision edges (MENTIONS, RELATES_TO, …) would intermix unrelated entities, so they
  are excluded.

- **Reproducibility: mandatory seed + deterministic ordering.** Louvain is stochastic;
  seed alone is insufficient (networkx result depends on iteration order), so nodes and
  edges enter sorted. Same graph + seed → same communities.

- **Communities in the same graph** as `COMMUNITY` nodes + `CONTAINS` edges (community→member),
  reusing the model. Summary and title in `Node.attrs` (`summary`/`title`) — no schema change
  (`attrs` already exists).

- **Isolation via pruned traversal projection, not disjoint indexing (option B).** A
  `COMMUNITY` node is a super-hub linked to all members; traversing `CONTAINS` would collapse all
  intra-community distance to 2 hops and leak the overlay into local retrieval. Fix: the
  `GraphStore` gains **generic** `exclude_node_types`/`exclude_edge_types` filter on
  `subgraph`/`neighbors`/`shortest_path` (the store **does not know** what COMMUNITY is — keeps the
  port swappable for Neo4j); the **policy** lives at the retrieval layer (`GraphRetriever`
  passes `{COMMUNITY}`/`{CONTAINS}`). `CommunityRetriever` does not traverse (anchors directly on
  summaries). Footgun of forgetting covered by topological invariance test, not by
  coupling the store.

- **Community id = stable hash (hashlib) of sorted members**, not index. Unchanged communities
  retain id between rebuilds → summarization (paid) skips unchanged ones.

- **Summarization boundary: logic in GLYPH, LLM injected.** `summarize_communities` assembles
  the prompt from members and populates `title`+`summary` via an injected `CommunitySummarizer`
  (Protocol) — provider/API key stay outside the lib (mirrors `DocumentExtractor`). AXON
  injects the model and fires on hook (sibling spec).

- **`CommunityRetriever` satisfies the `Retriever` port** (`retrieve(query, token_budget) ->
  ContextPack`, `mode="community"`), like every arm — drop-in where a `Retriever` is expected.

## Consequences

Global axis measurable with the same harness (global arm = follow-up, depends on a global
query set). Local retrieval is provably immune to the overlay. The store remains generic;
overlay policy is at the retrieval layer. Orchestration (hook, MCP tool `get_global_context`,
real model) is the sibling spec in AXON — outside this repo. Follow-ups: real summarization round
(costs LLM), global benchmark arm, and anchoring swap `summary`→`title` if benchmark
shows embedding dilution.
