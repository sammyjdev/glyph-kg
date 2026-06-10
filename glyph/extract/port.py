"""The Extractor port: turn a source into nodes and edges.

Two adapters will implement this in later phases: ``DocumentExtractor`` (LLM,
probabilistic) and ``CodeExtractor`` (tree-sitter, deterministic). The graph
core never knows which one produced a node — only the extraction differs.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from glyph.model.edge import Edge
from glyph.model.node import Node

Source = Path | str


@runtime_checkable
class Extractor(Protocol):
    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        """Read ``source`` and return the nodes and edges it yields."""
        ...
