"""HTTP transport for Groq/OpenAI-compat endpoints, plus a reference-free judge.

The transport (GROQ_BASE_URL, JsonPoster, _urllib_post) is shared with
generate.py so the answer generator reuses the same retry policy.
"""

from collections.abc import Callable
from typing import Any

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_MAX_RETRIES = 8
_BACKOFF_BASE_S = 2.0
_RETRY_CODES = (429, 500, 502, 503)

JsonPoster = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class JudgeError(Exception):
    """The judge call failed or its output was not the agreed JSON of metric scores."""


def _urllib_post(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float
) -> dict[str, Any]:  # pragma: no cover - real network wiring, exercised by the live test
    import json
    import ssl
    import time
    import urllib.error
    import urllib.request

    import certifi

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json", "User-Agent": "glyph-kg"},
        method="POST",
    )
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s, context=ssl_ctx) as response:
                body: dict[str, Any] = json.loads(response.read())
            return body
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_CODES or attempt == _MAX_RETRIES - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else _BACKOFF_BASE_S * (2**attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE_S * (2**attempt))
    raise JudgeError("judge endpoint exhausted retries")


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

_METRICS = ("faithfulness", "context_precision")


def _build_prompt(question: str, answer: str, contexts: list[str]) -> str:
    """Score faithfulness (grounded in contexts) and context_precision (contexts relevant)."""
    joined = "\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return (
        "You are evaluating a RAG (retrieval-augmented generation) system's answer.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved contexts:\n{joined}\n\n"
        f"Generated answer: {answer}\n\n"
        "Score the answer on two metrics, each from 0.0 to 1.0:\n\n"
        "- faithfulness: does every claim in the answer follow directly from the retrieved "
        "contexts, with no unsupported or contradicted statements? 1.0 = fully grounded, "
        "0.0 = entirely unsupported or contradicted.\n"
        "- context_precision: are the retrieved contexts relevant and necessary to answer "
        "the question? 1.0 = all contexts relevant, 0.0 = irrelevant.\n\n"
        'Respond with a single JSON object: {"faithfulness": <float>, "context_precision": <float>}'
    )


class OpenAICompatJudge:
    """Score faithfulness + context_precision in one call to an OpenAI-compatible endpoint.

    Reference-free (no expected answer needed) — works with Groq, NVIDIA NIM, or any
    OpenAI-compat provider via ``base_url``. No LLM framework dependency: a plain prompt
    and JSON response, scored and clamped to [0, 1] like the answer generator's transport.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = GROQ_BASE_URL,
        poster: JsonPoster | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._post = poster or _urllib_post
        self._timeout_s = timeout_s
        self._temperature = temperature

    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, float]:
        import json
        import re

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": _build_prompt(question, answer, contexts)}],
            "temperature": self._temperature,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = self._post(self._url, payload, headers, self._timeout_s)
        try:
            content = body["choices"][0]["message"]["content"] or ""
            # Some providers wrap JSON in ```json...``` code blocks — strip them.
            raw = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(raw.group(0) if raw else content)
            return {metric: max(0.0, min(1.0, float(parsed[metric]))) for metric in _METRICS}
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JudgeError(f"judge output not parseable: {exc}") from exc
