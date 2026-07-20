"""HTTP transport for Groq/OpenAI-compat endpoints, plus a reference-free judge.

The transport (GROQ_BASE_URL, JsonPoster, _urllib_post) is shared with
generate.py so the answer generator reuses the same retry policy.
"""

import re
from collections.abc import Callable
from typing import Any

from gnomon.domain.models import EvalCase, RagResponse
from gnomon.judge.ollama import JudgeProtocolError, parse_v1_judge_response
from gnomon.judge.prompts import build_prompt

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_MAX_RETRIES = 8
_BACKOFF_BASE_S = 2.0
_RETRY_CODES = (429, 500, 502, 503)
_ALLOWED_SCHEMES = frozenset({"http", "https"})

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
    from urllib.parse import urlparse

    import certifi

    scheme = urlparse(url).scheme
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"refusing to open url with disallowed scheme: {scheme!r}")

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above; ruff can't see that
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json", "User-Agent": "glyph-kg"},
        method="POST",
    )
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme checked above; ruff can't see that
                request, timeout=timeout_s, context=ssl_ctx
            ) as response:
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
        # EvalCase requires expected_answer/expected_contexts by schema, but the v1
        # judge prompt ignores them (reference-free judging, gnomon contract doc
        # section H) -- placeholders only satisfy gnomon's schema.
        case = EvalCase(id="_", question=question, expected_answer="_", expected_contexts=["_"])
        response = RagResponse(answer=answer, contexts=contexts, total_tokens=0, latency_ms=0.0)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": build_prompt(case, response)}],
            "temperature": self._temperature,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = self._post(self._url, payload, headers, self._timeout_s)
        try:
            content = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError(f"judge output not parseable: {exc}") from exc

        # Some providers wrap JSON in ```json...``` code blocks - strip them.
        raw = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            return dict(parse_v1_judge_response(raw.group(0) if raw else content).scores)
        except JudgeProtocolError as exc:
            raise JudgeError(f"judge output not parseable: {exc}") from exc
