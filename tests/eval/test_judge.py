"""OpenAICompatJudge speaks the OpenAI schema and reuses GNOMON's scoring contract."""

import json
import os
from typing import Any

import pytest

from glyph.eval.judge import GROQ_BASE_URL, JudgeError, OpenAICompatJudge

gnomon_models = pytest.importorskip("gnomon.domain.models")


def _case_and_response() -> tuple[Any, Any]:
    case = gnomon_models.EvalCase(
        id="q1",
        question="Quem é imune a fogo?",
        expected_answer="Balor",
        expected_contexts=["Balor — immune_to fogo"],
    )
    response = gnomon_models.RagResponse(
        answer="O balor é imune a fogo.",
        contexts=["Balor — immune_to fogo"],
        total_tokens=12,
        latency_ms=5.0,
    )
    return case, response


class FakePoster:
    """Records the request and returns a canned OpenAI-compatible chat completion."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.url: str | None = None
        self.payload: dict[str, Any] | None = None
        self.headers: dict[str, str] | None = None

    def __call__(self, url, payload, headers, timeout_s):  # type: ignore[no-untyped-def]
        self.url, self.payload, self.headers = url, payload, headers
        return {"choices": [{"message": {"content": self._content}}]}


def test_scores_parsed_and_keyed_by_v1_metrics() -> None:
    poster = FakePoster(json.dumps({"faithfulness": 0.9, "context_precision": 0.7}))
    judge = OpenAICompatJudge(model="llama-3.3-70b-versatile", api_key="k", poster=poster)
    case, response = _case_and_response()

    scores = judge.score(case, response, seed=7, run=0)

    assert scores.scores == {"faithfulness": 0.9, "context_precision": 0.7}


def test_request_targets_chat_completions_with_bearer_and_seed() -> None:
    poster = FakePoster(json.dumps({"faithfulness": 1.0, "context_precision": 1.0}))
    judge = OpenAICompatJudge(model="gpt-oss-20b", api_key="secret", poster=poster)
    case, response = _case_and_response()

    judge.score(case, response, seed=10, run=3)

    assert poster.url == GROQ_BASE_URL + "/chat/completions"
    assert poster.headers == {"Authorization": "Bearer secret"}
    assert poster.payload is not None
    assert poster.payload["model"] == "gpt-oss-20b"
    assert poster.payload["seed"] == 13  # seed + run
    assert poster.payload["response_format"] == {"type": "json_object"}


def test_seed_is_omitted_when_send_seed_false() -> None:
    # Some providers (Google Gemini) reject the `seed` field with HTTP 400.
    poster = FakePoster(json.dumps({"faithfulness": 1.0, "context_precision": 1.0}))
    judge = OpenAICompatJudge(model="gemini-2.5-flash", api_key="k", poster=poster, send_seed=False)
    case, response = _case_and_response()

    judge.score(case, response, seed=10, run=3)

    assert poster.payload is not None
    assert "seed" not in poster.payload
    assert poster.payload["response_format"] == {"type": "json_object"}


def test_scores_are_clamped_to_unit_interval() -> None:
    poster = FakePoster(json.dumps({"faithfulness": 1.4, "context_precision": -0.2}))
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)
    case, response = _case_and_response()

    scores = judge.score(case, response, seed=1, run=0)

    assert scores.scores == {"faithfulness": 1.0, "context_precision": 0.0}


def test_missing_metric_raises_judge_error() -> None:
    poster = FakePoster(json.dumps({"faithfulness": 0.5}))  # context_precision absent
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)
    case, response = _case_and_response()

    with pytest.raises(JudgeError):
        judge.score(case, response, seed=1, run=0)


def test_non_json_content_raises_judge_error() -> None:
    judge = OpenAICompatJudge(model="m", api_key="k", poster=FakePoster("not json"))
    case, response = _case_and_response()

    with pytest.raises(JudgeError):
        judge.score(case, response, seed=1, run=0)


@pytest.mark.live
def test_groq_live_smoke() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")
    judge = OpenAICompatJudge(model="llama-3.3-70b-versatile", api_key=api_key)
    case, response = _case_and_response()
    scores = judge.score(case, response, seed=1, run=0)
    assert set(scores.scores) == {"faithfulness", "context_precision"}
    assert all(0.0 <= v <= 1.0 for v in scores.scores.values())
