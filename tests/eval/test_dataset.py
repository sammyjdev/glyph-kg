"""Query set loads as Query instances with non-empty reference fields."""

import json
from pathlib import Path

from glyph.eval.dataset import Query, _expected_answer, load_eval_cases


def test_expected_answer_from_list_answer_key() -> None:
    assert (
        _expected_answer({"answer_key": ["fogo", "frio"], "relevant_labels": ["X"]}) == "fogo; frio"
    )


def test_expected_answer_from_string_answer_key() -> None:
    assert _expected_answer({"answer_key": "1/4", "relevant_labels": ["X"]}) == "1/4"


def test_expected_answer_falls_back_to_relevant_labels() -> None:
    assert (
        _expected_answer({"answer_key": None, "relevant_labels": ["Balor", "Vrock"]})
        == "Balor; Vrock"
    )


def test_load_eval_cases_maps_every_query(tmp_path: Path) -> None:
    queries = {
        "queries": [
            {
                "id": "q1",
                "question": "Quem resiste a fogo?",
                "answer_key": None,
                "relevant_labels": ["Balor"],
            },
            {
                "id": "q2",
                "question": "Qual o CR do aarakocra?",
                "answer_key": "1/4",
                "relevant_labels": ["Aarakocra"],
            },
        ]
    }
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(queries), encoding="utf-8")

    cases = load_eval_cases(path)

    assert [c.id for c in cases] == ["q1", "q2"]
    assert cases[0].reference_contexts == ["Balor"]
    assert cases[0].reference == "Balor"  # fallback
    assert cases[1].reference == "1/4"
    assert all(isinstance(c, Query) for c in cases)


def test_real_query_set_loads() -> None:
    cases = load_eval_cases(Path(__file__).resolve().parents[2] / "eval" / "queries.json")
    assert len(cases) == 25
    assert all(c.reference for c in cases)
    assert all(c.reference_contexts for c in cases)
