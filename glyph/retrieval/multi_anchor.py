"""P9: MultiAnchorRetriever — entity-aware multi-seed expansion.

Extracts named entities from the query text (CamelCase, acronyms, dec-/ADR- IDs)
and anchors each to its closest node, ensuring every explicitly named entity
gets representation in the subgraph — unlike GraphRetriever which uses the
whole-query embedding and may miss entities dominated by others.

Falls back to top-N embedding anchors when the query contains no named entities.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, count_tokens, pack
from glyph.model.edge import EdgeType
from glyph.model.graph import Subgraph
from glyph.model.node import Node, NodeType
from glyph.store.port import GraphStore

_OVERLAY_NODE_TYPES = frozenset({NodeType.COMMUNITY})
_OVERLAY_EDGE_TYPES = frozenset({EdgeType.CONTAINS})

# CamelCase / PascalCase words (≥ 4 chars after the leading capital) OR
# decision/ADR ID patterns like dec-121, ADR-G7.
_CAMEL_RE = re.compile(r"\b[A-Z]\w{3,}\b")
_ID_RE = re.compile(r"(?:dec|adr)-[\w-]+", re.IGNORECASE)


def _extract_named_entities(query: str) -> list[str]:
    """Return distinct named entities found in query, preserving first-seen order."""
    seen: set[str] = set()
    entities: list[str] = []
    for m in _ID_RE.finditer(query):
        e = m.group(0)
        if e not in seen:
            seen.add(e)
            entities.append(e)
    for m in _CAMEL_RE.finditer(query):
        e = m.group(0)
        if e not in seen:
            seen.add(e)
            entities.append(e)
    return entities


class MultiAnchorRetriever:
    """Expand subgraph from all named entities in the query simultaneously."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        nodes: Sequence[Node],
        hops: int = 2,
        anchors: int = 3,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._hops = hops
        self._anchors = anchors
        self._index = InMemoryVectorIndex()
        indexable = [n for n in nodes if n.type not in _OVERLAY_NODE_TYPES]
        self._n_nodes = len(indexable)
        if indexable:
            ids = [n.id for n in indexable]
            vectors = embedder.embed([n.label for n in indexable])
            for nid, vec in zip(ids, vectors, strict=True):
                self._index.add(nid, vec)

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        named = _extract_named_entities(query)
        seed: list[str] = []
        seen: set[str] = set()

        if named:
            # Anchor each named entity to its closest node.
            for entity in named:
                vec = self._embedder.embed([entity])[0]
                results = self._index.search(vec, 1)
                if results:
                    nid = results[0][0]
                    if nid not in seen:
                        seen.add(nid)
                        seed.append(nid)

        qvec = self._embedder.embed([query])[0]

        # Fallback: fill up to `anchors` with top-N by query embedding.
        if len(seed) < self._anchors:
            for nid, _ in self._index.search(qvec, self._anchors):
                if nid not in seen:
                    seen.add(nid)
                    seed.append(nid)

        if not seed:
            return pack("graph", [], token_budget, cost=count_tokens)

        all_scores = dict(self._index.search(qvec, self._n_nodes or 1))
        subgraph = self._store.subgraph(
            seed,
            self._hops,
            exclude_node_types=_OVERLAY_NODE_TYPES,
            exclude_edge_types=_OVERLAY_EDGE_TYPES,
        )
        return pack(
            "graph",
            self._segments(subgraph, set(seed), all_scores),
            token_budget,
            cost=count_tokens,
        )

    def _segments(
        self, subgraph: Subgraph, anchors: set[str], scores: dict[str, float]
    ) -> list[Segment]:
        label = {node.id: node.label for node in subgraph.nodes}
        out: dict[str, list[str]] = {}
        for edge in subgraph.edges:
            target = label.get(edge.dst, edge.dst)
            out.setdefault(edge.src, []).append(f"{edge.type.value} {target}")
        segments = []
        for node in subgraph.nodes:
            relations = "; ".join(out.get(node.id, []))
            text = f"{node.label} — {relations}" if relations else node.label
            score = 1.0 if node.id in anchors else scores.get(node.id, 0.0)
            segments.append(Segment(text=text, source=node.id, score=score))
        segments.sort(key=lambda s: (-s.score, s.source))
        return segments
