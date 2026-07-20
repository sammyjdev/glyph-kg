"""Property-based tests for the Markdown frontmatter stripper (security-gate #16).

# _strip_frontmatter is single-leading-block by contract, not idempotent — see #32.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from glyph.extract.document.markdown import _strip_frontmatter

_FRAGMENTS = st.sampled_from(["---\n", "  \n", "x\n", "\t---  \n"])
_ADVERSARIAL = st.lists(_FRAGMENTS, max_size=200).map("".join)


@given(text=st.one_of(st.text(max_size=2000), _ADVERSARIAL))
@settings(deadline=None)
def test_strip_frontmatter_never_raises_and_returns_str(text: str) -> None:
    result = _strip_frontmatter(text)
    assert isinstance(result, str)


@given(text=st.one_of(st.text(max_size=2000), _ADVERSARIAL))
@settings(deadline=None)
def test_strip_frontmatter_result_is_a_suffix_of_input(text: str) -> None:
    result = _strip_frontmatter(text)
    assert text.endswith(result)


@given(body=st.text(max_size=2000))
@settings(deadline=None)
def test_strip_frontmatter_identity_when_no_leading_delimiter(body: str) -> None:
    text = "x" + body
    assert _strip_frontmatter(text) == text


_INNER = st.text(max_size=50).filter(lambda s: "---" not in s)
_BODY = st.text(max_size=1000)


@given(inner=_INNER, body=_BODY)
@settings(deadline=None)
def test_strip_frontmatter_removes_exactly_one_leading_block(inner: str, body: str) -> None:
    text = f"---\n{inner}\n---\n{body}"
    assert _strip_frontmatter(text) == body


def test_strip_frontmatter_still_strips_trailing_horizontal_whitespace_on_closing_line() -> None:
    text = "---\nkey: v\n---  \t\nbody"
    assert _strip_frontmatter(text) == "body"


def test_strip_frontmatter_preserves_leading_blank_line_of_body() -> None:
    """Deterministic pin of the #33 counterexample — a hypothesis-sampled property
    test alone is not guaranteed to rediscover this exact input every run."""
    text = "---\n\n---\n\n"
    assert _strip_frontmatter(text) == "\n"
