"""The harness scores each arm via gnomon's run_eval, aggregating with bootstrap CIs."""

import pytest

from glyph.eval.dataset import Query
from glyph.eval.harness import run_benchmark, score_arm
from glyph.eval.response import ArmResponse

gnomon = pytest.importorskip("gnomon")


class FakeJudge:
    """Returns the same fixed scores for every case, regardless of content."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, str, list[str]]] = []

    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, float]:
        self.calls.append((question, answer, contexts))
        return dict(self._scores)


def _case(case_id: str) -> Query:
    return Query(id=case_id, question="Q?", reference="A", reference_contexts=["c"])


def _resp(input_tokens: int, output_tokens: int, latency: float) -> ArmResponse:
    return ArmResponse(
        answer="a",
        contexts=["c"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency,
    )


def test_score_arm_aggregates_metrics_and_cost() -> None:
    cases = [_case("q1"), _case("q2")]
    responses = {"q1": _resp(100, 20, 10.0), "q2": _resp(300, 40, 30.0)}
    judge = FakeJudge({"faithfulness": 0.8, "context_precision": 0.6})

    report = score_arm("graph", cases, responses, judge, seed=1)

    assert report.arm == "graph"
    assert report.n_cases == 2
    means = {m.metric: m.mean for m in report.metrics}
    assert means == {"faithfulness": 0.8, "context_precision": 0.6}
    assert report.total_tokens == 460  # 120 + 340
    assert report.mean_latency_ms == 20.0
    # input 400 @ $1/M + output 60 @ $5/M
    assert report.cost_usd == pytest.approx(400 / 1e6 + 60 * 5 / 1e6)
    # gnomon's EvalConfig.judge_runs floor is 2 (VAL-04): 2 cases * 2 runs.
    assert len(judge.calls) == 4


def test_score_arm_calls_judge_with_question_answer_contexts() -> None:
    cases = [_case("q1"), _case("q2")]
    responses = {"q1": _resp(0, 0, 0.0), "q2": _resp(0, 0, 0.0)}
    judge = FakeJudge({"faithfulness": 1.0})

    score_arm("graph", cases, responses, judge, seed=0)

    question, answer, contexts = judge.calls[0]
    assert question == "Q?"
    assert answer == "a"
    assert contexts == ["c"]


def test_score_arm_ci_bounds_mean_for_constant_scores() -> None:
    cases = [_case("q1"), _case("q2")]
    responses = {"q1": _resp(0, 0, 0.0), "q2": _resp(0, 0, 0.0)}
    judge = FakeJudge({"faithfulness": 0.9})

    report = score_arm("graph", cases, responses, judge, seed=0)

    stat = report.metrics[0]
    assert stat.ci_low <= stat.mean <= stat.ci_high


def test_score_arm_single_case_raises_value_error() -> None:
    """A single case cannot bound a population -- gnomon's aggregate_metric
    (called internally by run_eval) enforces n >= 2 cases; the harness no
    longer has a local degenerate fallback now that the custom loop is
    retired (ADR-G8 decision 3)."""
    cases = [_case("q1")]
    responses = {"q1": _resp(0, 0, 0.0)}
    judge = FakeJudge({"faithfulness": 0.9})

    with pytest.raises(ValueError, match="at least 2 cases"):
        score_arm("graph", cases, responses, judge, seed=0)


def test_score_arm_exposes_case_scores_for_reporting() -> None:
    """P3.5 (honest reporting) needs per-case scores to show where an arm
    lost. run_eval's EvalReport.case_scores must be threaded through to
    ArmReport, not dropped on the floor."""
    cases = [_case("q1"), _case("q2")]
    responses = {"q1": _resp(0, 0, 0.0), "q2": _resp(0, 0, 0.0)}
    judge = FakeJudge({"faithfulness": 0.8})

    report = score_arm("graph", cases, responses, judge, seed=0)

    assert set(report.case_scores.keys()) == {"faithfulness"}
    case_ids = {cs.case_id for cs in report.case_scores["faithfulness"]}
    assert case_ids == {"q1", "q2"}
    assert all(cs.score == 0.8 for cs in report.case_scores["faithfulness"])


def test_score_arm_delegates_to_gnomon_run_eval(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Scoring and aggregation must come from gnomon's run_eval (ADR-G8 decision
    3), not a private per-case loop -- proven by spying on the imported name."""
    from gnomon.domain.models import CaseCost, EvalReport, MetricResult

    import glyph.eval.harness as harness_mod

    calls: list[tuple[object, ...]] = []

    def fake_run_eval(cases, target, judge, config):  # type: ignore[no-untyped-def]
        calls.append((cases, target, judge, config))
        return EvalReport(
            metrics=[
                MetricResult(
                    metric="faithfulness",
                    mean=0.42,
                    ci_low=0.1,
                    ci_high=0.9,
                    n=2,
                    confidence_level=0.95,
                )
            ],
            per_case_cost=[CaseCost(case_id=c.id, total_tokens=0, latency_ms=0.0) for c in cases],
        )

    monkeypatch.setattr(harness_mod, "run_eval", fake_run_eval)

    cases = [_case("q1"), _case("q2")]
    responses = {"q1": _resp(0, 0, 0.0), "q2": _resp(0, 0, 0.0)}
    judge = FakeJudge({"faithfulness": 0.7})

    report = score_arm("graph", cases, responses, judge, seed=3)

    assert len(calls) == 1
    stat = report.metrics[0]
    assert stat.metric == "faithfulness"
    assert stat.mean == 0.42
    assert stat.ci_low == 0.1
    assert stat.ci_high == 0.9
    assert stat.n == 2


def test_run_benchmark_streams_each_arm_when_done() -> None:
    cases = [_case("q1"), _case("q2")]
    responses_by_arm = {
        "graph": {"q1": _resp(100, 10, 5.0), "q2": _resp(100, 10, 5.0)},
        "vector": {"q1": _resp(500, 10, 8.0), "q2": _resp(500, 10, 8.0)},
    }
    judge = FakeJudge({"faithfulness": 0.9, "context_precision": 0.5})
    done_arms: list[str] = []

    run_benchmark(
        cases,
        responses_by_arm,
        judge,
        judge_model="m",
        seed=0,
        on_arm=lambda r: done_arms.append(r.arm),
    )

    assert done_arms == ["graph", "vector"]


def test_run_benchmark_scores_every_arm() -> None:
    cases = [_case("q1"), _case("q2")]
    responses_by_arm = {
        "graph": {"q1": _resp(100, 10, 5.0), "q2": _resp(100, 10, 5.0)},
        "vector": {"q1": _resp(500, 10, 8.0), "q2": _resp(500, 10, 8.0)},
    }
    judge = FakeJudge({"faithfulness": 0.9})

    report = run_benchmark(cases, responses_by_arm, judge, judge_model="llama-3.3-70b", seed=3)

    assert report.n_cases == 2
    assert report.judge_model == "llama-3.3-70b"
    assert [a.arm for a in report.arms] == ["graph", "vector"]
    graph, vector = report.arms
    assert vector.total_tokens > graph.total_tokens


def test_run_benchmark_on_case_receives_arm_case_id_and_scores() -> None:
    cases = [_case("q1"), _case("q2")]
    responses_by_arm = {"graph": {"q1": _resp(0, 0, 0.0), "q2": _resp(0, 0, 0.0)}}
    judge = FakeJudge({"faithfulness": 0.7})
    received: list[tuple[str, str, dict[str, float]]] = []

    run_benchmark(
        cases,
        responses_by_arm,
        judge,
        judge_model="m",
        seed=0,
        on_case=lambda arm, case_id, scores: received.append((arm, case_id, scores)),
    )

    # gnomon's judge_runs floor is 2 (VAL-04): each case fires on_case once per run.
    assert received == [
        ("graph", "q1", {"faithfulness": 0.7}),
        ("graph", "q1", {"faithfulness": 0.7}),
        ("graph", "q2", {"faithfulness": 0.7}),
        ("graph", "q2", {"faithfulness": 0.7}),
    ]
