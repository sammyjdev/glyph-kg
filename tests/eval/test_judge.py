"""HTTP transport and OpenAICompatJudge: faithfulness + context_precision, no LLM framework."""

import json

from glyph.eval.judge import GROQ_BASE_URL, JudgeError, OpenAICompatJudge


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
    assert payload["response_format"] == {"type": "json_object"}


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
