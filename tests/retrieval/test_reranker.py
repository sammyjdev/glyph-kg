from glyph.model.contract import ContextPack, Segment
from glyph.retrieval.port import Retriever
from glyph.retrieval.reranked import RerankedRetriever
from glyph.retrieval.reranker import CrossEncoderReranker


class _FakeReranker:
    """Always returns segments in reverse order (last becomes first)."""

    def rerank(self, query: str, segments: list[Segment], k: int) -> list[Segment]:
        return list(reversed(segments))[:k]


class _FakeRetriever:
    """Returns a fixed list of segments, ignores token_budget."""

    def __init__(self, segments: list[Segment], last_budget: list[int] | None = None) -> None:
        self._segments = segments
        self._last_budget = last_budget if last_budget is not None else []

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        self._last_budget.append(token_budget)
        from glyph.model.contract import pack
        return pack("graph", self._segments, token_budget=50_000)


def _seg(text: str, score: float) -> Segment:
    return Segment(text=text, source=text, score=score)


def test_s01_reranker_sorts_by_relevance() -> None:
    """CrossEncoderReranker must rank the more relevant segment higher."""
    reranker = CrossEncoderReranker()
    segments = [
        _seg("Goblins are small green creatures that live in caves.", 0.5),
        _seg("The weather in Paris is often cloudy.", 0.5),
    ]
    result = reranker.rerank("Where do goblins live?", segments, k=2)
    assert result[0].source == "Goblins are small green creatures that live in caves."


def test_s01_reranker_truncates_to_k() -> None:
    """CrossEncoderReranker must return exactly k segments."""
    reranker = CrossEncoderReranker()
    segments = [_seg(f"text {i}", 0.5) for i in range(10)]
    result = reranker.rerank("query", segments, k=3)
    assert len(result) == 3


def test_s02_reranked_retriever_satisfies_port() -> None:
    """RerankedRetriever must satisfy the Retriever protocol."""
    retriever = _FakeRetriever([_seg("a", 1.0), _seg("b", 0.5)])
    reranker = _FakeReranker()
    wrapped = RerankedRetriever(retriever, reranker, k=2)
    assert isinstance(wrapped, Retriever)


def test_s03_reranked_uses_large_budget() -> None:
    """RerankedRetriever must call underlying retriever with budget>=50_000."""
    budgets: list[int] = []
    retriever = _FakeRetriever([_seg("x", 1.0)], last_budget=budgets)
    reranker = _FakeReranker()
    RerankedRetriever(retriever, reranker, k=1).retrieve("q", token_budget=1000)
    assert budgets[-1] >= 50_000


def test_s03_reranked_returns_k() -> None:
    """RerankedRetriever must return at most k segments in the pack."""
    segments = [_seg(f"text {i}" * 5, float(i)) for i in range(8)]
    retriever = _FakeRetriever(segments)
    reranker = _FakeReranker()
    pack = RerankedRetriever(retriever, reranker, k=3).retrieve("q", token_budget=10_000)
    assert len(pack.segments) <= 3
