"""P2.1: graph-aware retrieval — anchor a query, expand the neighborhood."""

from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, pack
from glyph.model.graph import Subgraph
from glyph.model.node import Node
from glyph.store.port import GraphStore


class GraphRetriever:
    """Embed node labels once; anchor a query to the nearest labels and expand by hops."""

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
        self._label = {node.id: node.label for node in nodes}
        self._index = InMemoryVectorIndex()
        ids = list(self._label)
        vectors = embedder.embed([self._label[node_id] for node_id in ids])
        for node_id, vector in zip(ids, vectors, strict=True):
            self._index.add(node_id, vector)

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        query_vector = self._embedder.embed([query])[0]
        anchors = [key for key, _ in self._index.search(query_vector, self._anchors)]
        subgraph = self._store.subgraph(anchors, self._hops)
        return pack("graph", self._segments(subgraph, set(anchors)), token_budget)

    def _segments(self, subgraph: Subgraph, anchors: set[str]) -> list[Segment]:
        label = {node.id: node.label for node in subgraph.nodes}
        out: dict[str, list[str]] = {}
        for edge in subgraph.edges:
            target = label.get(edge.dst, edge.dst)
            out.setdefault(edge.src, []).append(f"{edge.type.value} {target}")
        segments = []
        for node in subgraph.nodes:
            relations = "; ".join(out.get(node.id, []))
            text = f"{node.label} — {relations}" if relations else node.label
            score = 1.0 if node.id in anchors else 0.5
            segments.append(Segment(text=text, source=node.id, score=score))
        segments.sort(key=lambda s: (-s.score, s.source))  # source breaks score ties stably
        return segments
