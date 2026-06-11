"""AnswerGenerator maps contexts from segments and reports real summed token cost."""

import os

import pytest

from glyph.eval.generate import AnswerGenerator, AnthropicGenerator, Usage, build_prompt
from glyph.model.contract import ContextPack, Segment


class FakeRetriever:
    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, token_budget: int) -> ContextPack:
        self.calls.append((query, token_budget))
        return ContextPack(mode="graph", segments=self._segments, token_estimate=0)


class RecordingGenerator:
    def __init__(self, answer: str, usage: Usage) -> None:
        self._answer = answer
        self._usage = usage
        self.system: str | None = None
        self.prompt: str | None = None

    def generate(self, system: str, prompt: str) -> tuple[str, Usage]:
        self.system, self.prompt = system, prompt
        return self._answer, self._usage


def _segments() -> list[Segment]:
    return [
        Segment(text="Balor — immune_to fogo", source="balor", score=1.0),
        Segment(text="Vrock — resists frio", source="vrock", score=0.5),
    ]


def test_build_prompt_numbers_contexts_and_includes_question() -> None:
    prompt = build_prompt("Quem resiste a fogo?", ["Balor — resists fogo", "Vrock — resists frio"])
    assert "[1] Balor — resists fogo" in prompt
    assert "[2] Vrock — resists frio" in prompt
    assert "Quem resiste a fogo?" in prompt


def test_build_prompt_handles_empty_context() -> None:
    assert "sem contexto recuperado" in build_prompt("Q?", [])


def test_answer_maps_contexts_and_sums_tokens() -> None:
    retriever = FakeRetriever(_segments())
    generator = RecordingGenerator("Balor.", Usage(input_tokens=30, output_tokens=12))
    result = AnswerGenerator(retriever, generator, token_budget=500).answer("Quem é imune a fogo?")

    assert result.answer == "Balor."
    assert result.contexts == ["Balor — immune_to fogo", "Vrock — resists frio"]
    assert result.input_tokens == 30
    assert result.output_tokens == 12
    assert result.total_tokens == 42
    assert result.latency_ms >= 0.0
    assert retriever.calls == [("Quem é imune a fogo?", 500)]


def test_answer_feeds_retrieved_context_into_the_prompt() -> None:
    retriever = FakeRetriever(_segments())
    generator = RecordingGenerator("ok", Usage(input_tokens=1, output_tokens=1))
    AnswerGenerator(retriever, generator).answer("Q?")

    assert generator.prompt is not None
    assert "Balor — immune_to fogo" in generator.prompt
    assert "Q?" in generator.prompt


class _FakeBlock:
    def __init__(self, type_: str, text: str) -> None:
        self.type = type_
        self.text = text


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)

        class _Usage:
            input_tokens = 50
            output_tokens = 8

        class _Resp:
            content = [
                _FakeBlock("text", "Balor é imune a fogo."),
                _FakeBlock("tool_use", "ignored"),
            ]
            usage = _Usage()

        return _Resp()


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_anthropic_generator_concatenates_text_blocks_and_reports_usage() -> None:
    client = _FakeClient()
    text, usage = AnthropicGenerator(client=client).generate("sys", "prompt")
    assert text == "Balor é imune a fogo."  # tool_use block dropped
    assert usage == Usage(input_tokens=50, output_tokens=8)
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


@pytest.mark.live
def test_anthropic_generator_live_smoke() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    text, usage = AnthropicGenerator().generate(
        "Responda em uma palavra.", "Pergunta: qual a cor do céu num dia claro? Resposta:"
    )
    assert text.strip()
    assert usage.input_tokens + usage.output_tokens > 0
