"""P3.1/P3.3: score each arm's pre-computed answers with a judge, aggregate with bootstrap CIs.

Scores each case sequentially (one judge call per case) so progress streams live and
free-tier rate limits stay predictable. Per-case scores are preserved so the honest
report (P3.5) can show where each arm lost.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from gnomon.metrics.confidence import aggregate_metric

from glyph.eval.cost import total_cost_usd
from glyph.eval.dataset import Query
from glyph.eval.response import ArmResponse

ArmProgress = Callable[["ArmReport"], None]


class Judge(Protocol):
    """Scores one case's answer against its retrieved contexts."""

    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, float]: ...


@dataclass(frozen=True)
class MetricStat:
    """One metric's mean and bootstrap CI over the cases."""

    metric: str
    mean: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class ArmReport:
    """One retrieval arm's quality, token, latency and cost summary."""

    arm: str
    n_cases: int
    metrics: list[MetricStat]
    total_tokens: int
    mean_latency_ms: float
    cost_usd: float


@dataclass(frozen=True)
class BenchmarkReport:
    """The full comparison: every arm under one seed and metric set."""

    seed: int
    judge_model: str
    n_cases: int
    arms: list[ArmReport]


def _score_cases(
    cases: Sequence[Query],
    responses: Mapping[str, ArmResponse],
    judge: Judge,
    *,
    on_case: Callable[[str, dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    """Score each case sequentially — avoids concurrent-request limits on free-tier endpoints."""
    per_metric: dict[str, list[float]] = {}
    for case in cases:
        arm_resp = responses[case.id]
        case_scores = judge.score(case.question, arm_resp.answer, arm_resp.contexts)
        for metric, value in case_scores.items():
            per_metric.setdefault(metric, []).append(value)
        if on_case is not None:
            on_case(case.id, case_scores)
    return per_metric


def score_arm(
    arm: str,
    cases: Sequence[Query],
    responses: Mapping[str, ArmResponse],
    judge: Judge,
    *,
    seed: int = 0,
    on_case: Callable[[str, dict[str, float]], None] | None = None,
) -> ArmReport:
    """Score one arm: per-case judge call, aggregate per metric with bootstrap CIs."""
    ordered = [responses[case.id] for case in cases]
    per_metric = _score_cases(cases, responses, judge, on_case=on_case)

    metric_stats: list[MetricStat] = []
    for metric, values in sorted(per_metric.items()):
        if len(values) < 2:
            mean = fmean(values)
            ci_low = ci_high = mean
            n = len(values)
            metric_stats.append(
                MetricStat(metric=metric, mean=mean, ci_low=ci_low, ci_high=ci_high, n=n)
            )
        else:
            result = aggregate_metric(metric, values, confidence_level=0.95, seed=seed)
            metric_stats.append(
                MetricStat(
                    metric=result.metric,
                    mean=result.mean,
                    ci_low=result.ci_low,
                    ci_high=result.ci_high,
                    n=result.n,
                )
            )

    return ArmReport(
        arm=arm,
        n_cases=len(cases),
        metrics=metric_stats,
        total_tokens=sum(r.total_tokens for r in ordered),
        mean_latency_ms=fmean(r.latency_ms for r in ordered) if ordered else 0.0,
        cost_usd=total_cost_usd(ordered),
    )


def run_benchmark(
    cases: Sequence[Query],
    responses_by_arm: Mapping[str, Mapping[str, ArmResponse]],
    judge: Judge,
    *,
    judge_model: str,
    seed: int = 0,
    on_arm: ArmProgress | None = None,
    on_case: Callable[[str, str, dict[str, float]], None] | None = None,
) -> BenchmarkReport:
    """Score every arm over the same cases, judge and seed."""
    arms: list[ArmReport] = []
    for arm, responses in responses_by_arm.items():
        arm_on_case = (
            (lambda cid, sc, _arm=arm: on_case(_arm, cid, sc)) if on_case is not None else None
        )
        report = score_arm(arm, cases, responses, judge, seed=seed, on_case=arm_on_case)
        if on_arm is not None:
            on_arm(report)
        arms.append(report)
    return BenchmarkReport(seed=seed, judge_model=judge_model, n_cases=len(cases), arms=arms)
