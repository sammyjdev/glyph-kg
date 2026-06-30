"""P3.4/P3.5: serialize a BenchmarkReport, render METRICS.md, check for regression.

The honest report (P3.5) shows every arm's metric with its CI — including where the
graph did not win — plus token efficiency, cost and latency. The reproducible baseline
(P3.4) compares a fresh run's metric means against the committed numbers within a
tolerance, so a drift beyond noise fails the build.
"""

from collections.abc import Sequence
from typing import Any

from glyph.eval.harness import BenchmarkReport

DEFAULT_TOLERANCE = 0.05


def to_dict(report: BenchmarkReport) -> dict[str, Any]:
    """A stable, diffable JSON shape for the frozen baseline."""
    return {
        "seed": report.seed,
        "judge_model": report.judge_model,
        "n_cases": report.n_cases,
        "arms": [
            {
                "arm": arm.arm,
                "n_cases": arm.n_cases,
                "total_tokens": arm.total_tokens,
                "mean_latency_ms": round(arm.mean_latency_ms, 2),
                "cost_usd": round(arm.cost_usd, 6),
                "metrics": [
                    {
                        "metric": m.metric,
                        "mean": round(m.mean, 4),
                        "ci_low": round(m.ci_low, 4),
                        "ci_high": round(m.ci_high, 4),
                        "n": m.n,
                    }
                    for m in arm.metrics
                ],
            }
            for arm in report.arms
        ],
    }


def _metric_means(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    return {(arm["arm"], m["metric"]): m["mean"] for arm in payload["arms"] for m in arm["metrics"]}


def regression_check(
    committed: dict[str, Any], fresh: dict[str, Any], *, tolerance: float = DEFAULT_TOLERANCE
) -> list[str]:
    """Return human-readable violations where a fresh metric mean drifted past tolerance."""
    base = _metric_means(committed)
    new = _metric_means(fresh)
    violations: list[str] = []
    for key, ref in base.items():
        if key not in new:
            violations.append(f"{key[0]}/{key[1]}: missing from fresh run")
            continue
        delta = abs(new[key] - ref)
        if delta > tolerance:
            violations.append(
                f"{key[0]}/{key[1]}: {new[key]:.4f} vs committed {ref:.4f} "
                f"(|Δ|={delta:.4f} > {tolerance})"
            )
    return violations


def _metric_cell(metric: str, arms: Sequence[dict[str, Any]]) -> list[str]:
    cells = []
    for arm in arms:
        found = next((m for m in arm["metrics"] if m["metric"] == metric), None)
        if found is None:
            cells.append("—")
        else:
            cells.append(f"{found['mean']:.3f} [{found['ci_low']:.3f}, {found['ci_high']:.3f}]")
    return cells


def render_markdown(report: BenchmarkReport) -> str:
    """A reviewer-readable table: metrics with CIs, then token/cost/latency."""
    payload = to_dict(report)
    arms = payload["arms"]
    arm_names = [a["arm"] for a in arms]
    metric_names = sorted({m["metric"] for a in arms for m in a["metrics"]})

    header = "| Metric | " + " | ".join(arm_names) + " |"
    divider = "|" + "---|" * (len(arm_names) + 1)
    lines = [
        "# GLYPH benchmark — graph vs vector vs hybrid",
        "",
        f"- cases (n): **{payload['n_cases']}**  ·  judge: `{payload['judge_model']}`"
        f"  ·  seed: {payload['seed']}",
        "- metric cells show **mean [95% CI]** (percentile bootstrap, numpy, 2000 resamples).",
        "- cost is generation only (Haiku 4.5 rates); judge tokens excluded. Tokens are real.",
        "",
        header,
        divider,
    ]
    for metric in metric_names:
        lines.append(f"| {metric} | " + " | ".join(_metric_cell(metric, arms)) + " |")
    lines += [
        "| total tokens | " + " | ".join(str(a["total_tokens"]) for a in arms) + " |",
        "| cost (US$) | " + " | ".join(f"{a['cost_usd']:.4f}" for a in arms) + " |",
        "| mean latency (ms) | " + " | ".join(f"{a['mean_latency_ms']:.1f}" for a in arms) + " |",
        "",
    ]
    return "\n".join(lines)
