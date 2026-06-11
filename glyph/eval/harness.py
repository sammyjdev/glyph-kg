"""P3.1/P3.3: score each arm's pre-computed answers and aggregate with bootstrap CIs.

The harness bypasses GNOMON's ``run_eval`` on purpose: ``run_eval`` discards per-case
quality scores, and we want them (to report where the graph *lost*, P3.5). So we drive
the judge per case ourselves, collapse judge runs to one score per case, and reuse
GNOMON's ``aggregate_metric`` for the seeded percentile bootstrap — same CI machinery,
but we keep the per-case detail. Cost/latency come from the GLYPH-native ArmResponse.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Any, Protocol

from glyph.eval.cost import total_cost_usd
from glyph.eval.response import ArmResponse
from glyph.eval.target import to_rag_response

if TYPE_CHECKING:
    from gnomon.domain.models import EvalCase


class Judge(Protocol):
    """GNOMON's judge contract: score every v1 metric in one call per (case, run)."""

    def score(self, case: Any, response: Any, *, seed: int, run: int) -> Any: ...


@dataclass(frozen=True)
class MetricStat:
    """One metric's mean and bootstrap CI over the cases, as reported by GNOMON."""

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
    """The full comparison: every arm under one seed, judge and case set."""

    seed: int
    judge_runs: int
    judge_model: str
    n_cases: int
    arms: list[ArmReport]


def score_arm(
    arm: str,
    cases: Sequence["EvalCase"],
    responses: Mapping[str, ArmResponse],
    judge: Judge,
    *,
    seed: int,
    judge_runs: int,
) -> ArmReport:
    """Score one arm: judge each case ``judge_runs`` times, aggregate per metric."""
    from gnomon.metrics.confidence import aggregate_metric

    per_metric_case_scores: dict[str, list[float]] = {}
    ordered: list[ArmResponse] = []
    for case in cases:
        arm_response = responses[case.id]
        ordered.append(arm_response)
        rag_response = to_rag_response(arm_response)
        runs: dict[str, list[float]] = {}
        for run in range(judge_runs):
            scores = judge.score(case, rag_response, seed=seed, run=run).scores
            for metric, value in scores.items():
                runs.setdefault(metric, []).append(value)
        for metric, values in runs.items():
            per_metric_case_scores.setdefault(metric, []).append(fmean(values))

    metrics = [
        MetricStat(
            metric=result.metric,
            mean=result.mean,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            n=result.n,
        )
        for metric, case_scores in sorted(per_metric_case_scores.items())
        for result in [aggregate_metric(metric, case_scores, seed=seed)]
    ]
    return ArmReport(
        arm=arm,
        n_cases=len(cases),
        metrics=metrics,
        total_tokens=sum(r.total_tokens for r in ordered),
        mean_latency_ms=fmean(r.latency_ms for r in ordered) if ordered else 0.0,
        cost_usd=total_cost_usd(ordered),
    )


def run_benchmark(
    cases: Sequence["EvalCase"],
    responses_by_arm: Mapping[str, Mapping[str, ArmResponse]],
    judge: Judge,
    *,
    judge_model: str,
    seed: int = 0,
    judge_runs: int = 1,
) -> BenchmarkReport:
    """Score every arm over the same cases, judge and seed."""
    arms = [
        score_arm(arm, cases, responses, judge, seed=seed, judge_runs=judge_runs)
        for arm, responses in responses_by_arm.items()
    ]
    return BenchmarkReport(
        seed=seed,
        judge_runs=judge_runs,
        judge_model=judge_model,
        n_cases=len(cases),
        arms=arms,
    )
