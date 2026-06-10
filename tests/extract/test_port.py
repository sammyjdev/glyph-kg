from collections.abc import Sequence

from glyph.extract.port import Extractor, Source
from glyph.model.edge import Edge
from glyph.model.node import Node


class _Complete:
    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        return [], []


class _NotAnExtractor:
    def parse(self, source: Source) -> None: ...


def test_complete_implementation_satisfies_port() -> None:
    assert isinstance(_Complete(), Extractor)


def test_implementation_without_extract_does_not_satisfy_port() -> None:
    assert not isinstance(_NotAnExtractor(), Extractor)


def test_extract_returns_nodes_and_edges() -> None:
    nodes, edges = _Complete().extract("some/path")
    assert list(nodes) == []
    assert list(edges) == []
