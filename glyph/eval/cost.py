"""P3.3: USD cost of an arm's generation, from real token counts at Haiku 4.5 rates.

Only the generation step is billed here — retrieval is local embeddings (no tokens)
and the OSS judge is priced separately/negligibly. Generation input tokens are the
honest differentiator between arms (graph segments vs vector chunks drive context
size), so cost tracks token efficiency, which is the point of the comparison.
"""

from collections.abc import Iterable

from glyph.eval.response import ArmResponse

# Claude Haiku 4.5 pricing, USD per 1M tokens (matches glyph/extract/document/cost.py).
INPUT_PER_MILLION = 1.0
OUTPUT_PER_MILLION = 5.0


def response_cost_usd(arm: ArmResponse) -> float:
    """USD to generate one answer at the asymmetric input/output rates."""
    return (
        arm.input_tokens / 1_000_000 * INPUT_PER_MILLION
        + arm.output_tokens / 1_000_000 * OUTPUT_PER_MILLION
    )


def total_cost_usd(arms: Iterable[ArmResponse]) -> float:
    """USD to generate a whole arm's worth of answers."""
    return sum(response_cost_usd(arm) for arm in arms)
