"""Serialization, markdown rendering and regression check over a BenchmarkReport."""

from glyph.eval.harness import ArmReport, BenchmarkReport, MetricStat
from glyph.eval.report import regression_check, render_markdown, to_dict


def _report() -> BenchmarkReport:
    graph = ArmReport(
        arm="graph",
        n_cases=25,
        metrics=[
            MetricStat("context_precision", 0.81, 0.74, 0.88, 25),
            MetricStat("faithfulness", 0.90, 0.85, 0.95, 25),
        ],
        total_tokens=12000,
        mean_latency_ms=820.5,
        cost_usd=0.0123,
    )
    vector = ArmReport(
        arm="vector",
        n_cases=25,
        metrics=[
            MetricStat("context_precision", 0.62, 0.55, 0.69, 25),
            MetricStat("faithfulness", 0.88, 0.82, 0.93, 25),
        ],
        total_tokens=30000,
        mean_latency_ms=910.0,
        cost_usd=0.0301,
    )
    return BenchmarkReport(seed=0, judge_model="llama-3.3-70b", n_cases=25, arms=[graph, vector])


def test_to_dict_rounds_and_keeps_structure() -> None:
    payload = to_dict(_report())
    assert payload["n_cases"] == 25
    assert payload["arms"][0]["arm"] == "graph"
    assert payload["arms"][0]["metrics"][0]["mean"] == 0.81


def test_render_markdown_has_arms_and_metrics() -> None:
    md = render_markdown(_report())
    assert "graph" in md and "vector" in md
    assert "faithfulness" in md and "context_precision" in md
    assert "0.810 [0.740, 0.880]" in md  # graph context_precision cell
    assert "cost (US$)" in md


def test_regression_check_passes_within_tolerance() -> None:
    payload = to_dict(_report())
    assert regression_check(payload, payload) == []


def test_regression_check_flags_drift() -> None:
    committed = to_dict(_report())
    drifted = to_dict(_report())
    drifted["arms"][0]["metrics"][0]["mean"] = 0.50  # context_precision 0.81 -> 0.50
    violations = regression_check(committed, drifted, tolerance=0.05)
    assert len(violations) == 1
    assert "graph/context_precision" in violations[0]


def test_render_markdown_marks_absent_metric_with_dash() -> None:
    report = _report()
    # drop faithfulness from the vector arm so its cell renders as a dash
    vector = report.arms[1]
    report.arms[1] = ArmReport(
        arm=vector.arm,
        n_cases=vector.n_cases,
        metrics=[m for m in vector.metrics if m.metric != "faithfulness"],
        total_tokens=vector.total_tokens,
        mean_latency_ms=vector.mean_latency_ms,
        cost_usd=vector.cost_usd,
    )
    md = render_markdown(report)
    faithfulness_row = next(line for line in md.splitlines() if line.startswith("| faithfulness"))
    assert "—" in faithfulness_row


def test_regression_check_flags_missing_metric() -> None:
    committed = to_dict(_report())
    fresh = to_dict(_report())
    fresh["arms"][0]["metrics"] = []
    violations = regression_check(committed, fresh)
    assert any("missing" in v for v in violations)
