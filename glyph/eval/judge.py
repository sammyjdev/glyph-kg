"""P3.0: an OpenAI-compatible cloud judge (Groq default) implementing GNOMON's Judge.

GNOMON ships an OllamaJudge that POSTs the *Ollama* chat schema to a local host.
Cloud OSS providers (Groq, Together, Fireworks, OpenRouter) speak the *OpenAI*
schema instead — ``/chat/completions``, ``choices[0].message.content``, bearer auth.
This judge reuses GNOMON's exact scoring prompt (``gnomon.judge.prompts.build_prompt``)
and the same clamp/parse against ``V1_METRICS``, so a score is identical in meaning to
the Ollama judge — only the transport differs. It satisfies GNOMON's Judge protocol
(``score(case, response, *, seed, run) -> MetricScores``), so it drops into the harness.

The GNOMON import stays lazy: importing this module never requires the eval extra.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gnomon.domain.models import EvalCase, MetricScores, RagResponse

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# poster(url, payload, headers, timeout_s) -> decoded JSON body. Injected in tests.
JsonPoster = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class JudgeError(Exception):
    """The judge call failed or its output was not the agreed JSON of metric scores."""


def _urllib_post(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float
) -> dict[str, Any]:  # pragma: no cover - real network wiring, exercised by the live test
    import json
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise JudgeError(f"judge endpoint returned HTTP {response.status}")
        body: dict[str, Any] = json.loads(response.read())
    return body


class OpenAICompatJudge:
    """Score the v1 metrics in one call to an OpenAI-compatible OSS endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = GROQ_BASE_URL,
        poster: JsonPoster | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._post = poster or _urllib_post
        self._timeout_s = timeout_s
        self._temperature = temperature

    def score(
        self, case: "EvalCase", response: "RagResponse", *, seed: int, run: int
    ) -> "MetricScores":
        import json

        from gnomon.domain.models import MetricScores
        from gnomon.judge.prompts import build_prompt
        from gnomon.metrics.names import V1_METRICS

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": build_prompt(case, response)}],
            "response_format": {"type": "json_object"},
            "temperature": self._temperature,
            "seed": seed + run,  # deterministic sequence per declared seed, mirrors OllamaJudge
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = self._post(self._url, payload, headers, self._timeout_s)
        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            scores = {m: max(0.0, min(1.0, float(parsed[m]))) for m in V1_METRICS}
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JudgeError(f"judge output not parseable: {exc}") from exc
        return MetricScores(scores=scores)
