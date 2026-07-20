"""Property-based tests for _extract_named_entities (security-gate #16)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from glyph.retrieval.multi_anchor import _extract_named_entities

_TOKENS = st.sampled_from(["dec-121", "ADR-G7", "SomeCamelCaseWord", "dec-", "adr-x", ""])
_ADVERSARIAL = st.lists(st.one_of(_TOKENS, st.text(max_size=20)), max_size=50).map(" ".join)


@given(query=st.one_of(st.text(max_size=2000), _ADVERSARIAL))
@settings(deadline=None)
def test_extract_named_entities_never_raises_and_returns_list(query: str) -> None:
    result = _extract_named_entities(query)
    assert isinstance(result, list)


@given(query=st.one_of(st.text(max_size=2000), _ADVERSARIAL))
@settings(deadline=None)
def test_extract_named_entities_are_distinct(query: str) -> None:
    result = _extract_named_entities(query)
    assert len(result) == len(set(result))


@given(query=st.one_of(st.text(max_size=2000), _ADVERSARIAL))
@settings(deadline=None)
def test_extract_named_entities_are_substrings_of_query(query: str) -> None:
    result = _extract_named_entities(query)
    assert all(e in query for e in result)


def test_extract_named_entities_finds_known_id_and_camelcase() -> None:
    result = _extract_named_entities("see dec-121 and SomeCamelCaseWord for context")
    assert result == ["dec-121", "SomeCamelCaseWord"]
