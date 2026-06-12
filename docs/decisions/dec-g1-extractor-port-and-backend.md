# ADR-G1: Extractor port with two adapters and NetworkX/Neo4j backend

**Date:** 2026-06-09
**Status:** Accepted

## Context

GLYPH must build knowledge graphs across two domains with opposing extraction models:

- **Document**: probabilistic extraction. An LLM reads prose and infers entities and relations, with error. Example: "Goblin resists fire" is an inferred relation, not guaranteed.
- **Code**: deterministic extraction. tree-sitter produces the AST and the relation `A CALLS B` is fact, not inference.

Forcing both into a single generic extractor, or into a single graph schema serving both, produces an abstraction that excels at neither. The difference is not cosmetic: it is edge reliability.

In parallel, the library must be pip-installable and have good DX. A graph backend requiring a server (Neo4j) is high friction for a library. But "Neo4j" is a keyword that recruiters and job search scanners pick up on, and the production track record has value.

## Decision

**Extractor port with two adapters.** We define an `Extractor` protocol with `extract(source) -> (nodes, edges)`. `DocumentExtractor` (LLM) and `CodeExtractor` (tree-sitter) implement it. The graph core, store, and retrieval are shared; only extraction is domain-specific.

**GraphStore port with NetworkX default and Neo4j adapter.** NetworkX is the default backend: embedded, pip-installable, zero server, sufficient for target corpus scale (documents fit in memory; repositories do too). Neo4j exists as a smoke-tested adapter, for keyword and production history, not as an always-on service.

## Consequences

**Positive:**
- The correct boundary becomes explicit: what is common (graph, store, retrieval, measurement) vs. what is domain-specific (extraction). Demonstrates architectural abstraction, not just library use.
- Adding a third domain in the future is a new extractor adapter, with no core changes.
- Library DX preserved by NetworkX default; Neo4j keyword earned honestly because the adapter actually runs.

**Negative / trade-offs:**
- Maintaining two extraction adapters and two store adapters is a larger test surface.
- NetworkX does not scale for very large graphs; if a future corpus exceeds memory, the Neo4j adapter (or another) absorbs it, but that is additional work.

**Neutral / to observe:**
- The probabilistic/deterministic asymmetry is visible in the benchmark: document extraction quality is measured; code extraction is assumed correct within symbol resolution limitations.

## Alternatives considered

| Alternative | Why it was rejected |
|---|---|
| Single generic extractor | Domains have opposite edge reliability; a single extractor serves neither well |
| Single graph schema for both | Produces mediocre graphs in both domains; technical reviewer detects it |
| Neo4j as default | Requires a server, poor DX for a pip-installable library |
| kuzu as default | Embedded openCypher is attractive, but NetworkX has better DX and maturity for target scale; kuzu remains a future option if scale demands it |
| Separate projects (one doc, one code) | Loses the "KG library" claim; duplicates graph core, store, and measurement |
