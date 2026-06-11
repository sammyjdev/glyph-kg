"""Unified retrieval output: a budget-bounded bundle of scored text segments."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

Mode = Literal["graph", "vector", "hybrid"]

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count by characters (declared limitation; real tokenizer is Phase 3)."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


class Segment(BaseModel):
    """One unit of retrieved context, with where it came from and how relevant it scored."""

    model_config = ConfigDict(frozen=True)

    text: str
    source: str
    score: float


class ContextPack(BaseModel):
    """The comparable output of every retrieval arm."""

    model_config = ConfigDict(frozen=True)

    mode: Mode
    segments: list[Segment]
    token_estimate: int


def pack(mode: Mode, segments: Sequence[Segment], token_budget: int) -> ContextPack:
    """Take score-ordered segments up to ``token_budget`` (always at least the first)."""
    chosen: list[Segment] = []
    total = 0
    for segment in segments:
        cost = estimate_tokens(segment.text)
        if chosen and total + cost > token_budget:
            break
        chosen.append(segment)
        total += cost
    return ContextPack(mode=mode, segments=chosen, token_estimate=total)
