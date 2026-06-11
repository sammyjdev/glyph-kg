"""Local multilingual embedder (sentence-transformers), lazily imported."""

from collections.abc import Sequence

from glyph.embed.port import Vector

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbedder:
    """Wrap a SentenceTransformer model behind the Embedder port."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, model: object | None = None) -> None:
        if model is None:  # pragma: no cover - downloads weights, exercised by the slow test
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors = self._model.encode(list(texts))  # type: ignore[union-attr]
        return [list(map(float, vector)) for vector in vectors]
