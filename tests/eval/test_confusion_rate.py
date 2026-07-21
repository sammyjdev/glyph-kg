"""P3.5: attack-vs-resistance/immunity confusion class - error rate + rendering."""

import json
from pathlib import Path

import pytest

from glyph.eval.harness import ArmReport, BenchmarkReport, MetricStat
from glyph.eval.report import (
    attack_vs_resistance_rate,
    attack_vs_resistance_rate_from_file,
    render_markdown,
)

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_PATH = ROOT / "eval" / "queries.json"
NOTE_PATH = ROOT / "docs" / "decisions" / "phase3-p3.5-attack-vs-resistance.md"


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


def test_query_set_tags_discriminating_probes() -> None:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in payload["queries"]}
    assert by_id["ent-ankheg-resist"]["attack_vs_resistance_confusion"] is True

    probes_by_id = {p["id"]: p for p in payload["confusion_probes"]}
    assert probes_by_id["ent-espectro-resist"]["attack_vs_resistance_confusion"] is False
    assert probes_by_id["ent-deva-immune"]["attack_vs_resistance_confusion"] is False


def test_rate_is_nonzero_nonone_on_fixture() -> None:
    payload = {
        "queries": [
            {"id": "a", "attack_vs_resistance_confusion": True},
            {"id": "b", "attack_vs_resistance_confusion": True},
            {"id": "c", "attack_vs_resistance_confusion": False},
        ],
        "confusion_probes": [
            {"id": "d", "attack_vs_resistance_confusion": False},
            {"id": "e", "attack_vs_resistance_confusion": False},
        ],
    }
    assert attack_vs_resistance_rate(payload) == {
        "n_probes": 5,
        "confusions": 2,
        "error_rate": 0.4,
        "confused_ids": ["a", "b"],
    }


def test_rate_on_empty_payload_returns_zero_error_rate() -> None:
    # n_probes == 0: "no confusions measured" must read as 0.0, not the 1.0
    # a flipped fallback would silently produce.
    assert attack_vs_resistance_rate({"queries": [], "confusion_probes": []}) == {
        "n_probes": 0,
        "confusions": 0,
        "error_rate": 0.0,
        "confused_ids": [],
    }


def test_rate_ignores_untagged() -> None:
    payload = {
        "queries": [
            {"id": "a", "attack_vs_resistance_confusion": True},
            {"id": "b"},
        ],
        "confusion_probes": [
            {"id": "c"},
            {"id": "d", "attack_vs_resistance_confusion": False},
        ],
    }
    result = attack_vs_resistance_rate(payload)
    assert result["n_probes"] == 2
    assert result["confusions"] == 1
    assert result["confused_ids"] == ["a"]


def test_rate_on_real_query_set() -> None:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    result = attack_vs_resistance_rate(payload)
    assert result["n_probes"] == 3
    assert result["confusions"] == 1
    assert result["error_rate"] == pytest.approx(1 / 3)
    assert result["confused_ids"] == ["ent-ankheg-resist"]


def test_rate_from_file_reads_queries_json(tmp_path: Path) -> None:
    payload = {
        "queries": [
            {"id": "a", "attack_vs_resistance_confusion": True},
            {"id": "b", "attack_vs_resistance_confusion": True},
            {"id": "c", "attack_vs_resistance_confusion": False},
        ],
        "confusion_probes": [
            {"id": "d", "attack_vs_resistance_confusion": False},
            {"id": "e", "attack_vs_resistance_confusion": False},
        ],
    }
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(payload), encoding="utf-8")
    assert attack_vs_resistance_rate_from_file(queries_path) == {
        "n_probes": 5,
        "confusions": 2,
        "error_rate": 0.4,
        "confused_ids": ["a", "b"],
    }


def test_render_markdown_surfaces_rate() -> None:
    md = render_markdown(
        _report(),
        confusion={
            "n_probes": 3,
            "confusions": 1,
            "error_rate": 1 / 3,
            "confused_ids": ["ent-ankheg-resist"],
        },
    )
    assert "0.333" in md
    assert "attack-vs-resistance" in md.lower()
    assert "ent-ankheg-resist" in md


def test_render_markdown_omits_section_when_confusion_is_none() -> None:
    md = render_markdown(_report())
    assert "attack-vs-resistance" not in md.lower()


def test_followup_note_records_verdict() -> None:
    text = NOTE_PATH.read_text(encoding="utf-8")
    assert "does not justify" in text
    assert "ankheg" in text.lower()
    assert "1" in text
    assert "3" in text
