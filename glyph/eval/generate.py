"""P3.0: turn a retrieval arm into an answer, instrumented with real tokens + latency.

GNOMON's ``faithfulness`` scores how well the answer is grounded in the contexts, so
every arm needs a generation step over its retrieved context — not just the contexts.
The generator is grounded ("answer only from the context") so faithfulness is meaningful.
Tokens come from the model's usage; latency is wall-clock over retrieve + generate.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from glyph.eval.response import ArmResponse
from glyph.retrieval.port import Retriever

_SYSTEM = (
    "Você responde perguntas sobre criaturas do Monster Manual usando SOMENTE o "
    "contexto fornecido. Se o contexto não contiver a resposta, diga que não há "
    "informação suficiente. Seja conciso e não invente fatos fora do contexto."
)


@dataclass(frozen=True)
class Usage:
    """Token usage reported by the generation model."""

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class Generator(Protocol):
    """Produces an answer string and its token usage from a system + user prompt."""

    def generate(self, system: str, prompt: str) -> tuple[str, Usage]: ...


def build_prompt(question: str, contexts: list[str]) -> str:
    """Lay the retrieved context above the question, numbered for citeable grounding."""
    if contexts:
        joined = "\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    else:
        joined = "(sem contexto recuperado)"
    return f"Contexto:\n{joined}\n\nPergunta: {question}\nResposta:"


class AnswerGenerator:
    """Retrieve context for a question, generate a grounded answer, instrument the cost."""

    def __init__(
        self, retriever: Retriever, generator: Generator, token_budget: int = 1000
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._token_budget = token_budget

    def answer(self, question: str) -> ArmResponse:
        start = perf_counter()
        pack = self._retriever.retrieve(question, self._token_budget)
        contexts = [segment.text for segment in pack.segments]
        text, usage = self._generator.generate(_SYSTEM, build_prompt(question, contexts))
        latency_ms = (perf_counter() - start) * 1000.0
        return ArmResponse(
            answer=text,
            contexts=contexts,
            total_tokens=usage.total,
            latency_ms=latency_ms,
        )


class AnthropicGenerator:
    """Claude Haiku 4.5 generation over retrieved context, reporting real token usage."""

    def __init__(self, model: str = "claude-haiku-4-5", client: object | None = None) -> None:
        if client is None:  # pragma: no cover - real wiring, exercised by the live smoke test
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def generate(self, system: str, prompt: str) -> tuple[str, Usage]:
        response = self._client.messages.create(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return text, usage
