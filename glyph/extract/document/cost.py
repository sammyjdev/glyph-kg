"""P1.4: aggregate token usage into a USD cost report at Claude Haiku 4.5 rates."""

from collections.abc import Iterable
from dataclasses import dataclass

from glyph.extract.document.llm import Usage

# Claude Haiku 4.5 pricing, USD per 1M tokens.
INPUT_PER_MILLION = 1.0
OUTPUT_PER_MILLION = 5.0


@dataclass(frozen=True)
class CostReport:
    chunks: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def summarize(usages: Iterable[Usage]) -> CostReport:
    items = list(usages)
    input_tokens = sum(u.input_tokens for u in items)
    output_tokens = sum(u.output_tokens for u in items)
    cost = (
        input_tokens / 1_000_000 * INPUT_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_PER_MILLION
    )
    return CostReport(
        chunks=len(items),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 4),
    )
