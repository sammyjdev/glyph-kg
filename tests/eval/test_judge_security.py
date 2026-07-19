"""Security triage (ruff S310, issue #12): `_urllib_post` must refuse non-http(s) schemes.

New file — `tests/eval/test_judge.py` is not modified per the maker's no-touch rule.
"""

import pytest

from glyph.eval.judge import _urllib_post


def test_urllib_post_rejects_non_http_scheme_before_opening() -> None:
    with pytest.raises(ValueError, match="disallowed scheme"):
        _urllib_post("file:///etc/passwd", {}, {}, 1.0)


def test_urllib_post_accepts_https_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def _fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        calls.append((args, kwargs))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = _urllib_post("https://api.groq.com/openai/v1", {}, {}, 1.0)

    assert result == {}
    assert calls, "urlopen was never called; scheme guard likely blocked a valid https URL"
