from glyph.extract.document.cost import CostReport, summarize
from glyph.extract.document.llm import Usage


def test_summarize_sums_tokens_and_counts_chunks() -> None:
    report = summarize([Usage(1000, 200), Usage(2000, 300)])
    assert report.chunks == 2
    assert report.input_tokens == 3000
    assert report.output_tokens == 500


def test_summarize_prices_at_haiku_rates() -> None:
    # 1,000,000 input @ $1/M + 1,000,000 output @ $5/M = $6.00
    report = summarize([Usage(1_000_000, 1_000_000)])
    assert report.cost_usd == 6.0


def test_summarize_handles_empty() -> None:
    report = summarize([])
    assert report == CostReport(chunks=0, input_tokens=0, output_tokens=0, cost_usd=0.0)
