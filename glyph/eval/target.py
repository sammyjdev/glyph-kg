"""A pull-based gnomon target serving one arm's pre-computed responses.

Gnomon is pull-based -- ``run_eval`` calls ``target.query(question)``. GLYPH's
benchmark pre-computes each arm's :class:`ArmResponse` per question (one generation
pass, real tokens/latency) and serves them through one ``GlyphRagTarget`` per arm.
The gnomon import stays lazy so importing this module never requires the eval extra.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from glyph.eval.response import ArmResponse

if TYPE_CHECKING:
    from gnomon.domain.models import RagResponse


def to_rag_response(arm: ArmResponse) -> "RagResponse":
    """Map a GLYPH :class:`ArmResponse` onto gnomon's ``RagResponse`` (identical fields)."""
    from gnomon.domain.models import RagResponse

    return RagResponse(
        answer=arm.answer,
        contexts=arm.contexts,
        total_tokens=arm.total_tokens,
        latency_ms=arm.latency_ms,
    )


class GlyphRagTarget:
    """Serve pre-computed responses keyed by question text (satisfies gnomon's RagTarget)."""

    def __init__(self, responses: Mapping[str, ArmResponse]) -> None:
        self._responses = dict(responses)

    def query(self, question: str) -> Any:  # -> gnomon RagResponse (lazy import)
        try:
            arm = self._responses[question]
        except KeyError as exc:
            raise KeyError(f"no pre-computed response for question: {question!r}") from exc
        return to_rag_response(arm)
