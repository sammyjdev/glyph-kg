"""The harness scores each arm with a judge per case, aggregates with bootstrap CIs."""

import pytest

from glyph.eval.dataset import Query
from glyph.eval.harness import run_benchmark, score_arm
from glyph.eval.response import ArmResponse


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
    assert len(judge.calls) == 2


def test_score_arm_calls_judge_with_question_answer_contexts() -> None:
    cases = [_case("q1")]
    responses = {"q1": _resp(0, 0, 0.0)}
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
    cases = [_case("q1")]
    responses_by_arm = {"graph": {"q1": _resp(0, 0, 0.0)}}
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

    assert received == [("graph", "q1", {"faithfulness": 0.7})]
