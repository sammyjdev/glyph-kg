"""GLYPH-native arm response: the four fields GNOMON's RagResponse requires.

Kept independent of GNOMON so the generation/instrumentation core is testable
without the eval dependency installed. ``glyph.eval.target`` maps this onto the
real ``gnomon.domain.models.RagResponse`` (same field names and constraints).
"""

from pydantic import BaseModel, ConfigDict, Field


class ArmResponse(BaseModel):
    """One arm's answer to one question, with the real cost of producing it.

    Generation tokens are kept split (input vs output) because the model prices
    them asymmetrically; ``total_tokens`` is the sum GNOMON's RagResponse wants.
    """

    model_config = ConfigDict(frozen=True)

    answer: str
    contexts: list[str]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
