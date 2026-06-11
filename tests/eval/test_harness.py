"""The harness scores each arm per case and aggregates with GNOMON's bootstrap CI."""

import pytest

from glyph.eval.harness import run_benchmark, score_arm
from glyph.eval.response import ArmResponse

gnomon_models = pytest.importorskip("gnomon.domain.models")


class _Scores:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores


class FakeJudge:
    """Returns a fixed score per metric and records every call."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, int, int]] = []

    def score(self, case, response, *, seed: int, run: int):  # type: ignore[no-untyped-def]
        self.calls.append((case.id, seed, run))
        return _Scores(dict(self._scores))


def _case(case_id: str):  # type: ignore[no-untyped-def]
    return gnomon_models.EvalCase(
        id=case_id, question="Q?", expected_answer="A", expected_contexts=["c"]
    )


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

    report = score_arm("graph", cases, responses, judge, seed=1, judge_runs=2)

    assert report.arm == "graph"
    assert report.n_cases == 2
    means = {m.metric: m.mean for m in report.metrics}
    assert means == {"faithfulness": 0.8, "context_precision": 0.6}
    # constant scores -> CI collapses to the mean
    assert all(m.ci_low == m.mean == m.ci_high for m in report.metrics)
    assert report.total_tokens == 460  # 120 + 340
    assert report.mean_latency_ms == 20.0
    # input 400 @ $1/M + output 60 @ $5/M
    assert report.cost_usd == pytest.approx(400 / 1e6 + 60 * 5 / 1e6)
    # judge called n_cases * judge_runs times
    assert len(judge.calls) == 4


def test_score_arm_streams_per_case_progress() -> None:
    cases = [_case("q1"), _case("q2")]
    responses = {"q1": _resp(100, 20, 10.0), "q2": _resp(300, 40, 30.0)}
    judge = FakeJudge({"faithfulness": 0.8, "context_precision": 0.6})
    seen: list[tuple[str, int, int, str, dict[str, float]]] = []

    score_arm(
        "graph",
        cases,
        responses,
        judge,
        seed=1,
        judge_runs=2,
        on_case=lambda arm, done, total, cid, sc: seen.append((arm, done, total, cid, dict(sc))),
    )

    assert [(s[0], s[1], s[2], s[3]) for s in seen] == [
        ("graph", 1, 2, "q1"),
        ("graph", 2, 2, "q2"),
    ]
    assert seen[0][4] == {"faithfulness": 0.8, "context_precision": 0.6}


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
        judge_runs=1,
        on_arm=lambda report: done_arms.append(report.arm),
    )

    assert done_arms == ["graph", "vector"]


def test_run_benchmark_scores_every_arm() -> None:
    cases = [_case("q1"), _case("q2")]
    responses_by_arm = {
        "graph": {"q1": _resp(100, 10, 5.0), "q2": _resp(100, 10, 5.0)},
        "vector": {"q1": _resp(500, 10, 8.0), "q2": _resp(500, 10, 8.0)},
    }
    judge = FakeJudge({"faithfulness": 0.9, "context_precision": 0.5})

    report = run_benchmark(
        cases, responses_by_arm, judge, judge_model="llama-3.3-70b", seed=3, judge_runs=1
    )

    assert report.n_cases == 2
    assert report.judge_model == "llama-3.3-70b"
    assert [a.arm for a in report.arms] == ["graph", "vector"]
    # vector carries more context tokens -> higher cost, same quality here
    graph, vector = report.arms
    assert vector.total_tokens > graph.total_tokens
