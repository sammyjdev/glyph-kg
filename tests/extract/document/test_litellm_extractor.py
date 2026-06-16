"""Tests for LiteLLMExtractor and make_extractor factory."""

from __future__ import annotations

import json

import pytest

from glyph.extract.document.llm import (
    AnthropicExtractor,
    ExtractorConfig,
    LiteLLMExtractor,
    Usage,
    make_extractor,
)
from glyph.extract.document.schema import ExtractionResult


# ---------------------------------------------------------------------------
# Fake litellm client (duck-typed)
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt: int = 100, completion: int = 20) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeClient:
    """Injected as ``client`` into LiteLLMExtractor; records calls."""

    def __init__(self, response_content: str) -> None:
        self._content = response_content
        self.calls: list[dict] = []

    def completion(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(dict(kwargs))
        return _FakeResponse(self._content)


def _valid_json() -> str:
    result = ExtractionResult(entities=[], relations=[])
    return result.model_dump_json()


def _goblin_json() -> str:
    from glyph.extract.document.schema import ExtractedEntity, ExtractedRelation

    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Goblin", kind="creature"),
            ExtractedEntity(name="fire", kind="concept"),
        ],
        relations=[ExtractedRelation(subject="Goblin", predicate="RESISTS", object="fire")],
    )
    return result.model_dump_json()


# ---------------------------------------------------------------------------
# LiteLLMExtractor unit tests
# ---------------------------------------------------------------------------


def test_litellm_extractor_returns_parsed_result_and_usage() -> None:
    client = _FakeClient(_valid_json())
    extractor = LiteLLMExtractor(client=client)
    result, usage = extractor.extract("system", "text")
    assert isinstance(result, ExtractionResult)
    assert usage == Usage(input_tokens=100, output_tokens=20)


def test_litellm_extractor_passes_system_and_user_messages() -> None:
    client = _FakeClient(_valid_json())
    LiteLLMExtractor(client=client).extract("sys prompt", "user text")
    messages = client.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "sys prompt"}
    assert messages[1] == {"role": "user", "content": "user text"}


def test_litellm_extractor_passes_response_format() -> None:
    client = _FakeClient(_valid_json())
    LiteLLMExtractor(client=client).extract("s", "t")
    rf = client.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "ExtractionResult"


def test_litellm_extractor_parses_entities_and_relations() -> None:
    client = _FakeClient(_goblin_json())
    result, _ = LiteLLMExtractor(client=client).extract("s", "t")
    assert len(result.entities) == 2
    assert result.entities[0].name == "Goblin"
    assert result.relations[0].predicate == "RESISTS"


def test_litellm_extractor_retries_on_bad_json() -> None:
    """After failed parses the extractor raises ValueError."""
    client = _FakeClient("not valid json at all !!!")
    extractor = LiteLLMExtractor(client=client)
    with pytest.raises(ValueError, match="failed after"):
        extractor.extract("s", "t")


def test_litellm_extractor_default_model_is_ollama_llama3() -> None:
    client = _FakeClient(_valid_json())
    extractor = LiteLLMExtractor(client=client)
    assert extractor._model == "ollama/llama3"


def test_litellm_extractor_custom_model_forwarded() -> None:
    client = _FakeClient(_valid_json())
    extractor = LiteLLMExtractor(model="gemini/gemini-1.5-flash", client=client)
    extractor.extract("s", "t")
    assert client.calls[0]["model"] == "gemini/gemini-1.5-flash"


def test_litellm_extractor_satisfies_llm_extractor_protocol() -> None:
    """Structural check: LiteLLMExtractor has the ``extract`` signature."""
    from glyph.extract.document.llm import LLMExtractor

    client = _FakeClient(_valid_json())
    extractor = LiteLLMExtractor(client=client)
    # Protocol is not @runtime_checkable, so we check structurally.
    assert callable(extractor.extract)


# ---------------------------------------------------------------------------
# make_extractor factory tests
# ---------------------------------------------------------------------------


def test_make_extractor_default_returns_anthropic() -> None:
    extractor = make_extractor()
    assert isinstance(extractor, AnthropicExtractor)


def test_make_extractor_none_returns_anthropic() -> None:
    extractor = make_extractor(None)
    assert isinstance(extractor, AnthropicExtractor)


def test_make_extractor_anthropic_provider() -> None:
    cfg = ExtractorConfig(provider="anthropic")
    extractor = make_extractor(cfg)
    assert isinstance(extractor, AnthropicExtractor)


def test_make_extractor_litellm_provider() -> None:
    cfg = ExtractorConfig(provider="litellm", model="ollama/mistral")
    extractor = make_extractor(cfg)
    assert isinstance(extractor, LiteLLMExtractor)
    assert extractor._model == "ollama/mistral"


def test_make_extractor_litellm_forwards_base_url() -> None:
    cfg = ExtractorConfig(
        provider="litellm",
        model="ollama/llama3",
        base_url="http://localhost:11434",
    )
    extractor = make_extractor(cfg)
    assert isinstance(extractor, LiteLLMExtractor)
    assert extractor._base_url == "http://localhost:11434"


def test_make_extractor_litellm_notes_domain_sets_result_type() -> None:
    from glyph.extract.document.schema_notes import NotesExtractionResult

    cfg = ExtractorConfig(provider="litellm", model="ollama/llama3", domain="notes")
    extractor = make_extractor(cfg)
    assert isinstance(extractor, LiteLLMExtractor)
    assert extractor._result_type is NotesExtractionResult


def test_make_extractor_litellm_dnd_domain_sets_result_type() -> None:
    cfg = ExtractorConfig(provider="litellm", model="ollama/llama3", domain="dnd")
    extractor = make_extractor(cfg)
    assert isinstance(extractor, LiteLLMExtractor)
    assert extractor._result_type is ExtractionResult


def test_litellm_extractor_uses_result_type_for_schema_name() -> None:
    """The json_schema name in response_format matches the result_type class name."""
    from glyph.extract.document.schema_notes import NotesExtractionResult

    notes_json = NotesExtractionResult(entities=[], relations=[]).model_dump_json()
    client = _FakeClient(notes_json)
    extractor = LiteLLMExtractor(client=client, result_type=NotesExtractionResult)
    extractor.extract("s", "t")
    rf = client.calls[0]["response_format"]
    assert rf["json_schema"]["name"] == "NotesExtractionResult"


def test_litellm_extractor_with_notes_result_type_parses_notes_json() -> None:
    from glyph.extract.document.schema_notes import NotesExtractionResult, NotesEntity

    result = NotesExtractionResult(
        entities=[NotesEntity(name="Alice", kind="person")],
        relations=[],
    )
    client = _FakeClient(result.model_dump_json())
    extractor = LiteLLMExtractor(client=client, result_type=NotesExtractionResult)
    parsed, usage = extractor.extract("s", "t")
    assert isinstance(parsed, NotesExtractionResult)
    assert parsed.entities[0].name == "Alice"


def test_make_extractor_litellm_default_model() -> None:
    cfg = ExtractorConfig(provider="litellm")
    extractor = make_extractor(cfg)
    assert isinstance(extractor, LiteLLMExtractor)
    assert extractor._model == "ollama/llama3"


def test_make_extractor_unknown_provider_raises() -> None:
    cfg = ExtractorConfig(provider="unknown_provider")
    with pytest.raises(ValueError, match="Unknown provider"):
        make_extractor(cfg)


def test_make_extractor_anthropic_custom_model() -> None:
    cfg = ExtractorConfig(provider="anthropic", model="claude-haiku-4-5")
    extractor = make_extractor(cfg)
    assert isinstance(extractor, AnthropicExtractor)
    assert extractor._model == "claude-haiku-4-5"
