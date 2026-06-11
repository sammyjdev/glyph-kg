"""Generation cost from real token counts at the asymmetric Haiku 4.5 rates."""

from glyph.eval.cost import response_cost_usd, total_cost_usd
from glyph.eval.response import ArmResponse


def _arm(input_tokens: int, output_tokens: int) -> ArmResponse:
    return ArmResponse(
        answer="a",
        contexts=["c"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=1.0,
    )


def test_response_cost_uses_asymmetric_rates() -> None:
    # 1M input @ $1 + 1M output @ $5 = $6
    assert response_cost_usd(_arm(1_000_000, 1_000_000)) == 6.0


def test_response_cost_is_zero_without_tokens() -> None:
    assert response_cost_usd(_arm(0, 0)) == 0.0


def test_total_cost_sums_over_arm() -> None:
    arms = [_arm(500_000, 100_000), _arm(500_000, 100_000)]
    # input: 1M @ $1 = $1; output: 200k @ $5 = $1 -> $2
    assert total_cost_usd(arms) == 2.0
