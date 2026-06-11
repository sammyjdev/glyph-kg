"""P2.3: hybrid retrieval — fuse two retrievers by reciprocal rank fusion."""

from glyph.model.contract import ContextPack, Segment, pack
from glyph.retrieval.port import Retriever

_RRF_K = 60


class HybridRetriever:
    """Run a graph and a vector retriever and fuse their segments (injected, not imported)."""

    def __init__(self, graph: Retriever, vector: Retriever, rrf_k: int = _RRF_K) -> None:
        self._graph = graph
        self._vector = vector
        self._rrf_k = rrf_k

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        graph_pack = self._graph.retrieve(query, token_budget)
        vector_pack = self._vector.retrieve(query, token_budget)
        fused = self._fuse(graph_pack.segments, vector_pack.segments)
        return pack("hybrid", fused, token_budget)

    def _fuse(self, *arms: list[Segment]) -> list[Segment]:
        scores: dict[str, float] = {}
        first_seen: dict[str, Segment] = {}
        for segments in arms:
            for rank, segment in enumerate(segments):
                scores[segment.source] = scores.get(segment.source, 0.0) + 1.0 / (
                    self._rrf_k + rank + 1
                )
                first_seen.setdefault(segment.source, segment)
        merged = [
            Segment(text=first_seen[source].text, source=source, score=score)
            for source, score in scores.items()
        ]
        merged.sort(key=lambda s: -s.score)
        return merged
