"""P3.2: the frozen query set is well-formed and stays in sync with its generator."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_PATH = ROOT / "eval" / "queries.json"
GENERATOR_PATH = ROOT / "scripts" / "build_query_set.py"

_CATEGORIES = {
    "relational_single",
    "relational_multi",
    "entity_relation",
    "factual_attribute",
    "factual_description",
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("build_query_set", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_meta_count_matches_queries(payload: dict) -> None:
    assert payload["_meta"]["n"] == len(payload["queries"])


def test_ids_are_unique(payload: dict) -> None:
    ids = [q["id"] for q in payload["queries"]]
    assert len(ids) == len(set(ids))


def test_categories_are_known(payload: dict) -> None:
    assert {q["category"] for q in payload["queries"]} <= _CATEGORIES


def test_every_query_has_required_fields(payload: dict) -> None:
    for q in payload["queries"]:
        assert q["question"].strip()
        assert isinstance(q["graph_favored"], bool)
        assert isinstance(q["relevant_sources"], list)
        assert q["relevant_labels"] and len(q["relevant_labels"]) == len(q["relevant_sources"])


def test_relational_queries_have_relevant_sources(payload: dict) -> None:
    # A relational query the graph cannot answer (empty oracle) is a dead test case.
    for q in payload["queries"]:
        if q["category"].startswith("relational"):
            assert q["relevant_sources"], f"{q['id']} has no relevant sources"


def test_balanced_across_favored_hypotheses(payload: dict) -> None:
    favored = [q for q in payload["queries"] if q["graph_favored"]]
    against = [q for q in payload["queries"] if not q["graph_favored"]]
    assert favored and against, "the set must contain both graph- and vector-favored queries"


def test_committed_file_matches_generator(payload: dict, generator) -> None:
    graph = generator.Graph(
        json.loads((ROOT / "out" / "monster-manual.json").read_text(encoding="utf-8"))
    )
    assert generator.build(graph) == payload, "eval/queries.json is stale; regenerate it"
