"""RerankedRetriever: compose any Retriever with any Reranker."""

from glyph.model.contract import ContextPack, count_tokens, pack
from glyph.retrieval.port import Retriever
from glyph.retrieval.reranker import Reranker

_CANDIDATE_BUDGET = 50_000  # large enough to pass all candidates through


class RerankedRetriever:
    """Retrieve a large candidate set, then rerank with a cross-encoder to top-k."""

    def __init__(self, retriever: Retriever, reranker: Reranker, k: int = 5) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._k = k

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        candidates = self._retriever.retrieve(query, token_budget=_CANDIDATE_BUDGET)
        reranked = self._reranker.rerank(query, list(candidates.segments), self._k)
        return pack("reranked", reranked, token_budget, cost=count_tokens)
