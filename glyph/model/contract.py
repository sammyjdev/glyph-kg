"""Unified retrieval output: a budget-bounded bundle of scored text segments."""

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import tiktoken

Mode = Literal["graph", "vector", "hybrid", "community", "reranked"]

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count by characters (declared limitation; real tokenizer is Phase 3)."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


@functools.lru_cache(maxsize=1)
def _encoding() -> "tiktoken.Encoding":  # ponytail: lazy + cached; warms BPE vocab once per process
    import tiktoken

    # ponytail: relies on tiktoken's own network fetch on cold start; no built-in timeout knob found
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Real token count via tiktoken cl100k_base (Phase 3: real tokenizer where budget matters)."""
    # disallowed_special=(): retrieved text may legitimately contain special-token-looking
    # substrings (e.g. "<|endoftext|>"); tiktoken's default rejects those with ValueError.
    return len(_encoding().encode(text, disallowed_special=()))


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


def pack(
    mode: Mode,
    segments: Sequence[Segment],
    token_budget: int,
    cost: Callable[[str], int] = estimate_tokens,
) -> ContextPack:
    """Take score-ordered segments up to ``token_budget`` (always at least the first)."""
    chosen: list[Segment] = []
    total = 0
    for segment in segments:
        c = cost(segment.text)
        if chosen and total + c > token_budget:
            break
        chosen.append(segment)
        total += c
    return ContextPack(mode=mode, segments=chosen, token_estimate=total)
