from collections.abc import Sequence

from glyph.baseline.vector import VectorBaseline
from glyph.embed.port import Vector


class _FakeEmbedder:
    """Deterministic 3-dim keyword embedder: [fogo, caverna, goblin]."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> Vector:
        low = text.lower()
        return [
            1.0 if "fogo" in low else 0.0,
            1.0 if "caverna" in low else 0.0,
            1.0 if "goblin" in low else 0.0,
        ]


def _docs() -> list[tuple[str, str]]:
    return [
        ("Goblin", "O goblin resiste a fogo."),
        ("Orc", "O orc habita cavernas."),
    ]


def test_retrieve_ranks_relevant_chunk_first() -> None:
    baseline = VectorBaseline(embedder=_FakeEmbedder())
    baseline.index(_docs())
    pack = baseline.retrieve("fogo", token_budget=1000)
    assert pack.mode == "vector"
    assert pack.segments[0].source == "Goblin"
    assert "fogo" in pack.segments[0].text


def test_retrieve_returns_segments_with_chunk_text_and_source() -> None:
    baseline = VectorBaseline(embedder=_FakeEmbedder())
    baseline.index(_docs())
    pack = baseline.retrieve("caverna", token_budget=1000)
    top = pack.segments[0]
    assert top.source == "Orc"
    assert top.text == "O orc habita cavernas."


def test_retrieve_on_empty_index_returns_empty_pack() -> None:
    pack = VectorBaseline(embedder=_FakeEmbedder()).retrieve("fogo", token_budget=1000)
    assert pack.segments == []
    assert pack.mode == "vector"
