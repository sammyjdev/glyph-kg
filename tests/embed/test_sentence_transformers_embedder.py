import importlib.util

import pytest

from glyph.embed.port import Embedder
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder


class _FakeModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]


def test_embedder_delegates_to_the_model_and_satisfies_the_port() -> None:
    embedder = SentenceTransformerEmbedder(model=_FakeModel())
    assert isinstance(embedder, Embedder)
    vectors = embedder.embed(["a", "abc"])
    assert vectors == [[1.0, 1.0], [3.0, 1.0]]


@pytest.mark.slow
def test_real_model_ranks_similar_text_higher() -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")
    from glyph.embed.memory_index import InMemoryVectorIndex

    embedder = SentenceTransformerEmbedder()
    index = InMemoryVectorIndex()
    for key, text in [("fogo", "resistência a fogo"), ("frio", "imunidade a frio")]:
        index.add(key, embedder.embed([text])[0])
    top = index.search(embedder.embed(["dano de fogo"])[0], k=1)
    assert top[0][0] == "fogo"
