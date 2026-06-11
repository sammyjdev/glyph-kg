"""P3.0: turn a retrieval arm into an answer, instrumented with real tokens + latency.

GNOMON's ``faithfulness`` scores how well the answer is grounded in the contexts, so
every arm needs a generation step over its retrieved context — not just the contexts.
The generator is grounded ("answer only from the context") so faithfulness is meaningful.
Tokens come from the model's usage; latency is wall-clock over retrieve + generate.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from glyph.eval.judge import GROQ_BASE_URL, JsonPoster, _urllib_post
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
        self,
        retriever: Retriever,
        generator: Generator,
        token_budget: int = 1000,
        system: str = _SYSTEM,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._token_budget = token_budget
        self._system = system

    def answer(self, question: str) -> ArmResponse:
        start = perf_counter()
        pack = self._retriever.retrieve(question, self._token_budget)
        contexts = [segment.text for segment in pack.segments]
        text, usage = self._generator.generate(self._system, build_prompt(question, contexts))
        latency_ms = (perf_counter() - start) * 1000.0
        return ArmResponse(
            answer=text,
            contexts=contexts,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
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


class OpenAICompatGenerator:
    """Grounded generation via any OpenAI-compatible endpoint (NVIDIA NIM, Groq, ...).

    Lets the benchmark generate answers on a free OSS tier instead of paid Anthropic.
    The arms are still compared against each other under one generator, so the
    within-domain comparison stays fair; only the model identity differs from the
    document baseline (declared, not assumed). Reuses the judge's HTTP transport.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = GROQ_BASE_URL,
        poster: JsonPoster | None = None,
        timeout_s: float = 120.0,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._post = poster or _urllib_post
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> tuple[str, Usage]:
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = self._post(self._url, payload, headers, self._timeout_s)
        text: str = body["choices"][0]["message"]["content"]
        usage = Usage(
            input_tokens=body["usage"]["prompt_tokens"],
            output_tokens=body["usage"]["completion_tokens"],
        )
        return text, usage
