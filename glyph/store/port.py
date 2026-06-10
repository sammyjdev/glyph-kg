"""The GraphStore port: the contract every storage backend must satisfy."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from glyph.model.edge import Edge
from glyph.model.graph import NodeId, Path, Subgraph
from glyph.model.node import Node


@runtime_checkable
class GraphStore(Protocol):
    """Persist and traverse a graph of :class:`Node` and :class:`Edge`.

    Adapters (NetworkX default, Neo4j) implement this. Retrieval depends on the
    port, never on a concrete backend; swapping backends changes performance,
    not results.
    """

    def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        """Insert or update nodes, keyed by ``Node.id``."""
        ...

    def upsert_edges(self, edges: Sequence[Edge]) -> None:
        """Insert or update edges, keyed by ``(src, dst, type)``."""
        ...

    def neighbors(self, node: NodeId, hops: int) -> Subgraph:
        """Return the subgraph reachable within ``hops`` of ``node``."""
        ...

    def subgraph(self, seed: Sequence[NodeId], hops: int) -> Subgraph:
        """Return the subgraph reachable within ``hops`` of any seed node."""
        ...

    def shortest_path(self, src: NodeId, dst: NodeId) -> Path | None:
        """Return the shortest directed path, or ``None`` if unreachable."""
        ...
