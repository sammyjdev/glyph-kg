"""ArmResponse mirrors GNOMON's RagResponse constraints: non-negative tokens/latency."""

import pytest
from pydantic import ValidationError

from glyph.eval.response import ArmResponse


def test_builds_with_valid_fields_and_sums_total() -> None:
    r = ArmResponse(
        answer="a", contexts=["c1", "c2"], input_tokens=30, output_tokens=12, latency_ms=12.5
    )
    assert r.answer == "a"
    assert r.contexts == ["c1", "c2"]
    assert r.input_tokens == 30
    assert r.output_tokens == 12
    assert r.total_tokens == 42  # derived
    assert r.latency_ms == 12.5


def test_is_frozen() -> None:
    r = ArmResponse(answer="a", contexts=[], input_tokens=0, output_tokens=0, latency_ms=0.0)
    with pytest.raises(ValidationError):
        r.answer = "b"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value", [("input_tokens", -1), ("output_tokens", -1), ("latency_ms", -0.1)]
)
def test_rejects_negative_cost(field: str, value: float) -> None:
    kwargs = {
        "answer": "a",
        "contexts": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
    }
    kwargs[field] = value
    with pytest.raises(ValidationError):
        ArmResponse(**kwargs)
