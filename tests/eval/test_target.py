"""GlyphRagTarget serves pre-computed responses through gnomon's pull-based contract."""

import pytest

from glyph.eval.response import ArmResponse
from glyph.eval.target import GlyphRagTarget

gnomon_models = pytest.importorskip("gnomon.domain.models")
gnomon_interfaces = pytest.importorskip("gnomon.domain.interfaces")


def _arm() -> ArmResponse:
    return ArmResponse(
        answer="Balor.",
        contexts=["Balor — immune_to fogo"],
        input_tokens=6,
        output_tokens=4,
        latency_ms=3.0,
    )


def test_query_returns_mapped_rag_response() -> None:
    target = GlyphRagTarget({"Quem é imune a fogo?": _arm()})
    response = target.query("Quem é imune a fogo?")

    assert isinstance(response, gnomon_models.RagResponse)
    assert response.answer == "Balor."
    assert response.contexts == ["Balor — immune_to fogo"]
    assert response.total_tokens == 10
    assert response.latency_ms == 3.0


def test_satisfies_rag_target_protocol() -> None:
    target = GlyphRagTarget({"q": _arm()})
    assert isinstance(target, gnomon_interfaces.RagTarget)


def test_unknown_question_raises_keyerror() -> None:
    target = GlyphRagTarget({"known": _arm()})
    with pytest.raises(KeyError):
        target.query("unknown")
