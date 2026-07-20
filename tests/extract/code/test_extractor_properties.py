"""Property-based tests for CodeExtractor._parse (security-gate #16)."""

import pytest

pytest.importorskip("tree_sitter")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from glyph.extract.code.extractor import CodeExtractor  # noqa: E402
from glyph.extract.code.grammar import PythonGrammar  # noqa: E402

_GARBAGE_TEXT = st.text(max_size=2000).map(lambda s: s.encode("utf-8", "surrogatepass"))


@given(source=st.one_of(st.binary(max_size=4096), _GARBAGE_TEXT))
@settings(deadline=None)
def test_parse_never_raises_and_yields_a_tree(source: bytes) -> None:
    tree = CodeExtractor()._parse(PythonGrammar(), source)
    assert tree.root_node is not None
    # The tree must actually reflect the given input, not some fixed/ignored
    # constant: root span always covers exactly the full input length.
    assert tree.root_node.end_byte == len(source)
