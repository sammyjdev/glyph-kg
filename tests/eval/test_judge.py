"""HTTP transport and OpenAICompatJudge: faithfulness + context_precision, no LLM framework."""

import json

import pytest

from glyph.eval.judge import GROQ_BASE_URL, JudgeError, OpenAICompatJudge

gnomon = pytest.importorskip("gnomon")


def test_groq_base_url_points_to_openai_compat_endpoint() -> None:
    assert GROQ_BASE_URL.endswith("/v1")


def test_judge_error_is_exception() -> None:
    assert issubclass(JudgeError, Exception)


def _poster(content: str):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def poster(url: str, payload: dict, headers: dict, timeout_s: float) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"content": content}}]}

    return poster, captured


def test_score_posts_to_chat_completions_with_bearer_auth() -> None:
    poster, captured = _poster(json.dumps({"faithfulness": 0.9, "context_precision": 0.7}))
    judge = OpenAICompatJudge(model="llama-3.3-70b-versatile", api_key="secret", poster=poster)

    scores = judge.score(
        "Quem é imune a fogo?", "O balor é imune a fogo.", ["Balor — immune_to fogo"]
    )

    assert scores == {"faithfulness": 0.9, "context_precision": 0.7}
    assert captured["url"] == GROQ_BASE_URL + "/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    payload = captured["payload"]
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert "response_format" not in payload


def test_score_includes_question_answer_and_contexts_in_prompt() -> None:
    poster, captured = _poster(json.dumps({"faithfulness": 1.0, "context_precision": 1.0}))
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)

    judge.score("Quem é imune a fogo?", "O balor é imune a fogo.", ["Balor — immune_to fogo"])

    prompt = captured["payload"]["messages"][0]["content"]  # type: ignore[index]
    assert "Quem é imune a fogo?" in prompt
    assert "O balor é imune a fogo." in prompt
    assert "Balor — immune_to fogo" in prompt


def test_scores_are_clamped_to_unit_interval() -> None:
    poster, _ = _poster(json.dumps({"faithfulness": 1.4, "context_precision": -0.2}))
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)

    scores = judge.score("q", "a", ["c"])

    assert scores == {"faithfulness": 1.0, "context_precision": 0.0}


def test_unparseable_output_raises_judge_error() -> None:
    poster, _ = _poster("not json")
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)

    try:
        judge.score("q", "a", ["c"])
    except JudgeError:
        return
    raise AssertionError("expected JudgeError")


def test_missing_metric_in_output_raises_judge_error() -> None:
    poster, _ = _poster(json.dumps({"faithfulness": 0.5}))  # context_precision missing
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)

    try:
        judge.score("q", "a", ["c"])
    except JudgeError:
        return
    raise AssertionError("expected JudgeError")


def test_score_prompt_equals_gnomon_build_prompt() -> None:
    """ADR-G8 (a): the prompt sent to the endpoint must be gnomon's canonical
    build_prompt output, byte for byte -- not a locally duplicated prompt that
    can silently drift from gnomon's semantics."""
    from gnomon.domain.models import EvalCase, RagResponse
    from gnomon.judge.prompts import build_prompt

    poster, captured = _poster(json.dumps({"faithfulness": 1.0, "context_precision": 1.0}))
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)

    judge.score("Quem é imune a fogo?", "O balor é imune a fogo.", ["Balor — immune_to fogo"])

    expected = build_prompt(
        EvalCase(
            id="_",
            question="Quem é imune a fogo?",
            expected_answer="_",
            expected_contexts=["_"],
        ),
        RagResponse(
            answer="O balor é imune a fogo.",
            contexts=["Balor — immune_to fogo"],
            total_tokens=0,
            latency_ms=0.0,
        ),
    )
    assert captured["payload"]["messages"][0]["content"] == expected  # type: ignore[index]


def test_score_parse_delegates_to_gnomon_parse_v1_judge_response(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """ADR-G8 (a): parsing must be delegated to gnomon's parse_v1_judge_response,
    not a locally reimplemented parse -- proven by spying on the imported name."""
    from gnomon.domain.models import MetricScores

    import glyph.eval.judge as judge_mod

    calls: list[str] = []

    def fake_parse(content: str) -> MetricScores:
        calls.append(content)
        return MetricScores(scores={"faithfulness": 0.55, "context_precision": 0.66})

    monkeypatch.setattr(judge_mod, "parse_v1_judge_response", fake_parse)

    poster, _ = _poster(json.dumps({"ignored-by-fake": True}))
    judge = OpenAICompatJudge(model="m", api_key="k", poster=poster)

    scores = judge.score("q", "a", ["c"])

    assert calls, "parse_v1_judge_response was never called"
    assert scores == {"faithfulness": 0.55, "context_precision": 0.66}
