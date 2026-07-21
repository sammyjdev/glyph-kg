"""P2.1: graph-aware retrieval — anchor a query, expand the neighborhood."""

from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, count_tokens, pack
from glyph.model.edge import EdgeType
from glyph.model.graph import Subgraph
from glyph.model.node import Node, NodeType
from glyph.store.port import GraphStore

# The community overlay (P7) is a global-axis construct; local neighborhood expansion
# must never traverse it, or COMMUNITY super-hubs collapse structural distances (dec-g7).
_OVERLAY_NODE_TYPES = frozenset({NodeType.COMMUNITY})
_OVERLAY_EDGE_TYPES = frozenset({EdgeType.CONTAINS})


class GraphRetriever:
    """Embed node labels once; anchor a query to the nearest labels and expand by hops."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        nodes: Sequence[Node],
        hops: int = 2,
        anchors: int = 3,
        pagerank_weight: float = 0.0,
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
        # Pre-compute normalized PageRank (max=1.0) once at index time.
        self._pagerank: dict[str, float] = {}
        if pagerank_weight > 0.0:
            raw = store.pagerank()
            max_pr = max(raw.values(), default=1.0)
            self._pagerank = {k: v / max_pr for k, v in raw.items()}
        self._pagerank_weight = pagerank_weight

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        query_vector = self._embedder.embed([query])[0]
        anchors = [key for key, _ in self._index.search(query_vector, self._anchors)]
        all_scores = dict(self._index.search(query_vector, len(self._label)))
        subgraph = self._store.subgraph(
            anchors,
            self._hops,
            exclude_node_types=_OVERLAY_NODE_TYPES,
            exclude_edge_types=_OVERLAY_EDGE_TYPES,
        )
        return pack(
            "graph",
            self._segments(subgraph, set(anchors), all_scores),
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
            if node.id in anchors:
                score = 1.0
            else:
                cosine = scores.get(node.id, 0.0)
                pr = self._pagerank.get(node.id, 0.0)
                # Linear blend confirmed correct for this case (Perplexity research, 2026-07-01):
                # RRF is for fusing independent retriever rank lists, not for mixing a semantic
                # similarity score with a structural prior like centrality — score-level fusion
                # is the right tool here.
                score = (1.0 - self._pagerank_weight) * cosine + self._pagerank_weight * pr
            segments.append(Segment(text=text, source=node.id, score=score))
        segments.sort(key=lambda s: (-s.score, s.source))  # source breaks score ties stably
        return segments
