"""P3.1/P3.3: score each arm's pre-computed answers via gnomon's run_eval.

Per ADR-G8 decision 3, the custom per-case scoring loop is retired: each arm's
pre-computed ArmResponses (P3.0) are served through GlyphRagTarget and scored with
gnomon's run_eval, which owns both the per-case judge loop and the cross-case
bootstrap aggregation (already used directly for aggregation alone, ADR-G8 (b)).
Per-case scores are preserved via EvalReport.case_scores so the honest report
(P3.5) can still show where each arm lost.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Protocol

from gnomon.config.config import EvalConfig
from gnomon.domain.models import EvalCase, MetricScores, RagResponse
from gnomon.runner.runner import run_eval

from glyph.eval.cost import total_cost_usd
from glyph.eval.dataset import Query
from glyph.eval.response import ArmResponse
from glyph.eval.target import GlyphRagTarget

# Judge-run floor gnomon's EvalConfig enforces (VAL-04): a bootstrap CI needs at
# least 2 runs to denoise. OpenAICompatJudge is deterministic (temperature=0), so
# both runs score identically -- the second call is the price of the shared
# aggregation layer, not new information.
_JUDGE_RUNS = 2

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
class CaseScore:
    """One case's denoised score for one metric -- GLYPH's mirror of gnomon's
    CaseScore, exposed so the honest report (P3.5) can show where an arm lost."""

    case_id: str
    score: float


@dataclass(frozen=True)
class ArmReport:
    """One retrieval arm's quality, token, latency and cost summary."""

    arm: str
    n_cases: int
    metrics: list[MetricStat]
    total_tokens: int
    mean_latency_ms: float
    cost_usd: float
    case_scores: dict[str, list[CaseScore]] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkReport:
    """The full comparison: every arm under one seed and metric set."""

    seed: int
    judge_model: str
    n_cases: int
    arms: list[ArmReport]


class _GnomonJudgeAdapter:
    """Adapts GLYPH's simple Judge (question/answer/contexts) to gnomon's Judge
    protocol (case/response/seed/run). Also forwards each (case, run) score to an
    optional on_case callback -- run_eval has no per-case hook of its own, so this
    adapter is where P3.1's live-progress streaming still happens.
    """

    def __init__(
        self, judge: Judge, *, on_case: Callable[[str, dict[str, float]], None] | None
    ) -> None:
        self._judge = judge
        self._on_case = on_case

    def score(self, case: EvalCase, response: RagResponse, *, seed: int, run: int) -> MetricScores:
        scores = self._judge.score(case.question, response.answer, response.contexts)
        if self._on_case is not None:
            self._on_case(case.id, scores)
        return MetricScores(scores=scores)


def score_arm(
    arm: str,
    cases: Sequence[Query],
    responses: Mapping[str, ArmResponse],
    judge: Judge,
    *,
    seed: int = 0,
    on_case: Callable[[str, dict[str, float]], None] | None = None,
) -> ArmReport:
    """Score one arm: gnomon's run_eval drives the per-case judge loop and
    cross-case aggregation. ArmResponses were already generated (P3.0), so
    GlyphRagTarget only replays them.
    """
    ordered = [responses[case.id] for case in cases]
    eval_cases = [
        EvalCase(
            id=case.id,
            question=case.question,
            expected_answer=case.reference,
            expected_contexts=case.reference_contexts,
        )
        for case in cases
    ]
    target = GlyphRagTarget({case.question: responses[case.id] for case in cases})
    gnomon_judge = _GnomonJudgeAdapter(judge, on_case=on_case)
    config = EvalConfig(seed=seed, judge_runs=_JUDGE_RUNS, confidence_level=0.95)

    report = run_eval(eval_cases, target, gnomon_judge, config)

    metric_stats = [
        MetricStat(metric=m.metric, mean=m.mean, ci_low=m.ci_low, ci_high=m.ci_high, n=m.n)
        for m in report.metrics
    ]
    case_scores = {
        metric: [CaseScore(case_id=cs.case_id, score=cs.score) for cs in scores]
        for metric, scores in report.case_scores.items()
    }

    return ArmReport(
        arm=arm,
        n_cases=len(cases),
        metrics=metric_stats,
        total_tokens=sum(r.total_tokens for r in ordered),
        mean_latency_ms=fmean(r.latency_ms for r in ordered) if ordered else 0.0,
        cost_usd=total_cost_usd(ordered),
        case_scores=case_scores,
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
