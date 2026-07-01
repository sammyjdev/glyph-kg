"""Cross-encoder reranker: rescores candidate segments query-by-query."""

from typing import Protocol, runtime_checkable

from glyph.model.contract import Segment

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, segments: list[Segment], k: int) -> list[Segment]: ...


class CrossEncoderReranker:
    """Rerank segments using a sentence-transformers CrossEncoder model (local, no API)."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model)

    def rerank(self, query: str, segments: list[Segment], k: int) -> list[Segment]:
        if not segments:
            return []
        pairs = [(query, s.text) for s in segments]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, segments, strict=True), key=lambda x: -float(x[0]))
        return [
            Segment(text=s.text, source=s.source, score=float(score))
            for score, s in ranked[:k]
        ]
