from glyph.model.contract import ContextPack, Segment
from glyph.retrieval.hybrid import HybridRetriever
from glyph.retrieval.port import Retriever


class _CannedRetriever:
    def __init__(self, mode, segments: list[Segment]) -> None:
        self._mode = mode
        self._segments = segments

    def retrieve(self, query: str, token_budget: int) -> ContextPack:
        return ContextPack(mode=self._mode, segments=self._segments, token_estimate=0)


def test_hybrid_satisfies_the_port() -> None:
    graph = _CannedRetriever("graph", [])
    vector = _CannedRetriever("vector", [])
    assert isinstance(HybridRetriever(graph, vector), Retriever)


def test_hybrid_fuses_and_dedupes_by_source() -> None:
    graph = _CannedRetriever(
        "graph",
        [Segment(text="A", source="a", score=1.0), Segment(text="B", source="b", score=0.9)],
    )
    vector = _CannedRetriever(
        "vector",
        [Segment(text="B", source="b", score=0.8), Segment(text="C", source="c", score=0.7)],
    )
    result = HybridRetriever(graph, vector).retrieve("q", token_budget=1000)
    assert result.mode == "hybrid"
    sources = [s.source for s in result.segments]
    assert sources.count("b") == 1  # deduped across arms
    assert set(sources) == {"a", "b", "c"}
    assert sources[0] == "b"  # appears in both arms -> highest fused rank


def test_hybrid_respects_token_budget() -> None:
    graph = _CannedRetriever("graph", [Segment(text="x" * 40, source="g", score=1.0)])
    vector = _CannedRetriever("vector", [Segment(text="y" * 40, source="v", score=1.0)])
    result = HybridRetriever(graph, vector).retrieve("q", token_budget=10)
    assert len(result.segments) == 1  # only one 10-token segment fits
