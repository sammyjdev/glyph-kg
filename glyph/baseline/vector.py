"""P2.2: a fair vector baseline over the same per-creature chunk texts."""

from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder, VectorIndex
from glyph.model.contract import ContextPack, Segment, pack


class VectorBaseline:
    """Embed chunk texts, retrieve top-k by cosine, return a ContextPack."""

    def __init__(self, embedder: Embedder, index: VectorIndex | None = None) -> None:
        self._embedder = embedder
        self._index = index if index is not None else InMemoryVectorIndex()
        self._text: dict[str, str] = {}

    def index(self, documents: Sequence[tuple[str, str]]) -> None:
        """Index ``(source_label, text)`` documents (e.g. one per creature chunk)."""
        labels = [label for label, _ in documents]
        texts = [text for _, text in documents]
        for label, text, vector in zip(labels, texts, self._embedder.embed(texts), strict=True):
            self._index.add(label, vector)
            self._text[label] = text

    def retrieve(self, query: str, token_budget: int = 1000, k: int = 20) -> ContextPack:
        query_vector = self._embedder.embed([query])[0]
        hits = self._index.search(query_vector, k)
        segments = [Segment(text=self._text[key], source=key, score=score) for key, score in hits]
        return pack("vector", segments, token_budget)
