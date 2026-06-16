"""LLM adapters for entity/relation extraction: Anthropic structured-output and LiteLLM.

``AnthropicExtractor`` calls ``messages.parse`` with the extraction schema (structured
output, no retry needed).  ``LiteLLMExtractor`` routes to any provider supported by
litellm (Ollama, OpenRouter, Gemini, …) using JSON-mode + Pydantic validation with a
small retry on parse failure.  Both satisfy the ``LLMExtractor`` Protocol.

Use ``make_extractor(cfg)`` to select the right implementation from config.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from glyph.extract.document.schema import ExtractionResult

_log = logging.getLogger(__name__)

# Maximum re-parse attempts for ``LiteLLMExtractor`` when JSON is malformed.
_MAX_RETRIES = 2


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


class LLMExtractor(Protocol):
    def extract(self, system: str, text: str) -> tuple[ExtractionResult, Usage]: ...


class AnthropicExtractor:
    """Calls ``messages.parse`` with the extraction schema and reports token usage."""

    def __init__(self, model: str = "claude-haiku-4-5", client: object | None = None) -> None:
        if client is None:  # pragma: no cover - real wiring, exercised by the live smoke test
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def extract(self, system: str, text: str) -> tuple[ExtractionResult, Usage]:
        response = self._client.messages.parse(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": text}],
            output_format=ExtractionResult,
        )
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.parsed_output, usage


class LiteLLMExtractor:
    """Provider-agnostic extractor via *litellm*.

    ``model`` accepts any litellm model string, e.g.:
    - ``"ollama/llama3"``              (local Ollama, default base_url omitted)
    - ``"openrouter/deepseek/deepseek-chat"``
    - ``"gemini/gemini-1.5-flash"``
    - ``"anthropic/claude-haiku-4-5"``
    - ``"nvidia_nim/meta/llama3-70b-instruct"``

    Structured output is requested via ``response_format`` carrying the
    ``ExtractionResult`` JSON schema.  On parse failure the raw content is
    validated with Pydantic; if that also fails, up to ``_MAX_RETRIES``
    additional attempts are made before raising.
    """

    def __init__(
        self,
        model: str = "ollama/llama3",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,  # injectable for tests (duck-typed completion callable)
        max_tokens: int = 8192,
        result_type: type[BaseModel] | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = client  # None → import litellm and call litellm.completion
        self._max_tokens = max_tokens
        # result_type controls which Pydantic model the JSON response is validated against.
        # Defaults to ExtractionResult (D&D domain); override with NotesExtractionResult
        # for the notes/Obsidian domain.
        self._result_type: type[BaseModel] = (
            result_type if result_type is not None else ExtractionResult
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _completion(self, messages: list[dict[str, str]], response_format: Any) -> Any:
        """Call litellm.completion (or injected client) and return the raw response."""
        if self._client is not None:
            return self._client.completion(
                model=self._model,
                messages=messages,
                response_format=response_format,
            )
        import litellm  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "response_format": response_format,
        }
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        return litellm.completion(**kwargs)

    def _parse_content(self, content: str) -> BaseModel:
        """Parse raw JSON string from the model into ``self._result_type``."""
        data = json.loads(content)
        return self._result_type.model_validate(data)

    # ------------------------------------------------------------------
    # Public interface (LLMExtractor Protocol)
    # ------------------------------------------------------------------

    def extract(self, system: str, text: str) -> tuple[Any, Usage]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        schema_name = self._result_type.__name__
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": self._result_type.model_json_schema(),
            },
        }

        last_exc: Exception | None = None
        for attempt in range(1 + _MAX_RETRIES):
            try:
                response = self._completion(messages, response_format)
                content: str = response.choices[0].message.content or ""
                result = self._parse_content(content)
                raw_usage = response.usage
                usage = Usage(
                    input_tokens=int(raw_usage.prompt_tokens),
                    output_tokens=int(raw_usage.completion_tokens),
                )
                return result, usage
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_exc = exc
                _log.warning("LiteLLMExtractor parse attempt %d failed: %s", attempt + 1, exc)

        raise ValueError(f"LiteLLMExtractor failed after {1 + _MAX_RETRIES} attempts") from last_exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass
class ExtractorConfig:
    """Selects and configures an ``LLMExtractor`` implementation.

    provider="anthropic"  → ``AnthropicExtractor``
    provider="litellm"    → ``LiteLLMExtractor``

    domain="dnd"          → D&D extraction schema (default)
    domain="notes"        → personal-notes / Obsidian schema
    """

    provider: str = "anthropic"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    domain: str = "dnd"
    extra: dict[str, Any] = field(default_factory=dict)


def make_extractor(cfg: ExtractorConfig | None = None) -> LLMExtractor:
    """Return the appropriate ``LLMExtractor`` from *cfg* (defaults to Anthropic)."""
    if cfg is None:
        return AnthropicExtractor()
    if cfg.provider == "anthropic":
        kwargs: dict[str, Any] = {}
        if cfg.model is not None:
            kwargs["model"] = cfg.model
        return AnthropicExtractor(**kwargs)
    if cfg.provider == "litellm":
        model = cfg.model if cfg.model is not None else "ollama/llama3"
        result_type: type[BaseModel] = ExtractionResult
        if cfg.domain == "notes":
            from glyph.extract.document.schema_notes import (  # noqa: PLC0415
                NotesExtractionResult,
            )

            result_type = NotesExtractionResult
        return LiteLLMExtractor(
            model=model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            result_type=result_type,
            **cfg.extra,
        )
    raise ValueError(f"Unknown provider: {cfg.provider!r}. Expected 'anthropic' or 'litellm'.")
