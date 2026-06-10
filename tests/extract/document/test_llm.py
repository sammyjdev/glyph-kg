import os

import pytest

from glyph.extract.document.llm import AnthropicExtractor, Usage
from glyph.extract.document.schema import ExtractionResult


class _FakeMessages:
    def __init__(self, parsed: ExtractionResult) -> None:
        self._parsed = parsed
        self.calls: list[dict] = []

    def parse(self, **kwargs: object):
        self.calls.append(kwargs)

        class _Usage:
            input_tokens = 100
            output_tokens = 20

        class _Resp:
            parsed_output = self._parsed
            usage = _Usage()

        return _Resp()


class _FakeClient:
    def __init__(self, parsed: ExtractionResult) -> None:
        self.messages = _FakeMessages(parsed)


def test_anthropic_extractor_returns_parsed_output_and_usage() -> None:
    parsed = ExtractionResult(entities=[], relations=[])
    client = _FakeClient(parsed)
    extractor = AnthropicExtractor(client=client)
    result, usage = extractor.extract("system text", "verbete")
    assert result is parsed
    assert usage == Usage(input_tokens=100, output_tokens=20)


def test_anthropic_extractor_uses_haiku_by_default() -> None:
    client = _FakeClient(ExtractionResult(entities=[], relations=[]))
    AnthropicExtractor(client=client).extract("s", "t")
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_anthropic_extractor_requests_generous_output_budget() -> None:
    client = _FakeClient(ExtractionResult(entities=[], relations=[]))
    AnthropicExtractor(client=client).extract("s", "t")
    assert client.messages.calls[0]["max_tokens"] == 8192


@pytest.mark.live
def test_anthropic_extractor_live_smoke() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from glyph.extract.document.prompt import system_prompt

    extractor = AnthropicExtractor()
    text = "GOBLIN\nO goblin é um humanoide pequeno que resiste a fogo e habita cavernas."
    result, usage = extractor.extract(system_prompt(), text)
    assert usage.input_tokens > 0
    assert any(r.predicate == "RESISTS" for r in result.relations)
