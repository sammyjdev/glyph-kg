"""P8: PathRetriever — shortest-path arm.

For queries like 'what connects X to Y?' or decision-lineage questions.
Anchors to the two most similar nodes, finds shortest_path, renders the
walk as scored segments with edge-type annotations.
"""
from __future__ import annotations

from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, pack
from glyph.model.edge import EdgeType
from glyph.model.node import Node, NodeType
from glyph.store.port import GraphStore

_OVERLAY_NODE_TYPES = frozenset({NodeType.COMMUNITY})
_OVERLAY_EDGE_TYPES = frozenset({EdgeType.CONTAINS})


class PathRetriever:
    """Anchor query to two nodes; return the shortest structural path between them."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        nodes: Sequence[Node],
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._label: dict[str, str] = {}
        self._index = InMemoryVectorIndex()
        # Index only non-overlay nodes — community summaries are not path endpoints.
        indexable = [n for n in nodes if n.type not in _OVERLAY_NODE_TYPES]
        if indexable:
            ids = [n.id for n in indexable]
            vectors = embedder.embed([n.label for n in indexable])
            for nid, vec in zip(ids, vectors, strict=True):
                self._index.add(nid, vec)
            self._label = {n.id: n.label for n in indexable}

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        qvec = self._embedder.embed([query])[0]
        top2 = [key for key, _ in self._index.search(qvec, 2)]

        if len(top2) < 2:
            segments = [
                Segment(text=self._label.get(nid, nid), source=nid, score=1.0)
                for nid in top2
            ]
            return pack("graph", segments, token_budget)

        src, dst = top2[0], top2[1]
        path = self._store.shortest_path(
            src, dst,
            exclude_node_types=_OVERLAY_NODE_TYPES,
            exclude_edge_types=_OVERLAY_EDGE_TYPES,
        )

        if path is None:
            segments = [
                Segment(text=self._label.get(src, src), source=src, score=1.0),
                Segment(text=self._label.get(dst, dst), source=dst, score=1.0),
            ]
            return pack("graph", segments, token_budget)

        # Fetch edges between consecutive path nodes to annotate the walk.
        subgraph = self._store.subgraph(
            path.nodes, hops=0,
            exclude_node_types=_OVERLAY_NODE_TYPES,
            exclude_edge_types=_OVERLAY_EDGE_TYPES,
        )
        # Build adjacency: (src_id, dst_id) -> edge_type for quick lookup.
        adj: dict[tuple[str, str], str] = {
            (e.src, e.dst): e.type.value for e in subgraph.edges
        }

        n = len(path.nodes)
        segments: list[Segment] = []
        for i, nid in enumerate(path.nodes):
            label = self._label.get(nid, nid)
            # Annotate with the edge to the next node if present.
            if i < n - 1:
                next_id = path.nodes[i + 1]
                rel = adj.get((nid, next_id), "→")
                text = f"{label} —[{rel}]→"
            else:
                text = label
            score = 1.0 if i in (0, n - 1) else 0.7
            segments.append(Segment(text=text, source=nid, score=score))

        return pack("graph", segments, token_budget)
