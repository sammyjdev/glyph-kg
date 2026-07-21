"""P3.4/P3.5: serialize a BenchmarkReport, render METRICS.md, check for regression.

The honest report (P3.5) shows every arm's metric with its CI — including where the
graph did not win — plus token efficiency, cost and latency. The reproducible baseline
(P3.4) compares a fresh run's metric means against the committed numbers within a
tolerance, so a drift beyond noise fails the build.
"""

import json
from collections.abc import Sequence
from pathlib import Path
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


def attack_vs_resistance_rate(payload: dict[str, Any]) -> dict[str, Any]:
    """P3.5: error rate for the attack-vs-resistance/immunity confusion class.

    Combines tagged entries from the main query set with the dedicated
    `confusion_probes` ground-truth contrasts - both carry the same
    `attack_vs_resistance_confusion` key, so a probe is IN the class if the
    key is present anywhere in either source, regardless of value.
    """
    tagged = [q for q in payload.get("queries", []) if "attack_vs_resistance_confusion" in q]
    tagged += [
        q for q in payload.get("confusion_probes", []) if "attack_vs_resistance_confusion" in q
    ]
    confused = [q for q in tagged if q["attack_vs_resistance_confusion"]]
    n = len(tagged)
    return {
        "n_probes": n,
        "confusions": len(confused),
        "error_rate": (len(confused) / n) if n else 0.0,
        "confused_ids": sorted(q["id"] for q in confused),
    }


def attack_vs_resistance_rate_from_file(queries_path: str | Path) -> dict[str, Any]:
    """Load the raw queries.json payload and compute the P3.5 confusion rate from it."""
    payload = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    return attack_vs_resistance_rate(payload)


def render_markdown(report: BenchmarkReport, *, confusion: dict[str, Any] | None = None) -> str:
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
    if confusion is not None and confusion["n_probes"] > 0:
        confused_ids = ", ".join(f"`{cid}`" for cid in confusion["confused_ids"])
        lines += [
            "## Attack-vs-resistance confusion (P3.5)",
            "",
            f"- discrimination probes: {confusion['n_probes']} · confirmed extraction "
            f"confusions: {confusion['confusions']} · error rate: "
            f"{confusion['error_rate']:.3f}",
            f"- confused cases: {confused_ids}",
            "",
        ]
    return "\n".join(lines)
