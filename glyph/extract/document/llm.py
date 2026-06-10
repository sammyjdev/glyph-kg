"""Thin Anthropic adapter: Claude Haiku 4.5 structured-output extraction + token usage."""

from dataclasses import dataclass
from typing import Protocol

from glyph.extract.document.schema import ExtractionResult


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
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": text}],
            output_format=ExtractionResult,
        )
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.parsed_output, usage
