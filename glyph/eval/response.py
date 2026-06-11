"""GLYPH-native arm response: the four fields GNOMON's RagResponse requires.

Kept independent of GNOMON so the generation/instrumentation core is testable
without the eval dependency installed. ``glyph.eval.target`` maps this onto the
real ``gnomon.domain.models.RagResponse`` (same field names and constraints).
"""

from pydantic import BaseModel, ConfigDict, Field


class ArmResponse(BaseModel):
    """One arm's answer to one question, with the real cost of producing it."""

    model_config = ConfigDict(frozen=True)

    answer: str
    contexts: list[str]
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
