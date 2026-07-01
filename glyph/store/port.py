"""The GraphStore port: the contract every storage backend must satisfy."""

from collections.abc import Collection, Sequence
from typing import Protocol, runtime_checkable

from glyph.model.edge import Edge, EdgeType
from glyph.model.graph import NodeId, Path, Subgraph
from glyph.model.node import Node, NodeType


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

    def neighbors(
        self,
        node: NodeId,
        hops: int,
        *,
        exclude_node_types: Collection[NodeType] = frozenset(),
        exclude_edge_types: Collection[EdgeType] = frozenset(),
    ) -> Subgraph:
        """Return the subgraph reachable within ``hops`` of ``node``."""
        ...

    def subgraph(
        self,
        seed: Sequence[NodeId],
        hops: int,
        *,
        exclude_node_types: Collection[NodeType] = frozenset(),
        exclude_edge_types: Collection[EdgeType] = frozenset(),
    ) -> Subgraph:
        """Return the subgraph reachable within ``hops`` of any seed node.

        ``exclude_node_types``/``exclude_edge_types`` prune the traversal graph
        before expansion — the retrieval layer passes the community overlay
        (``COMMUNITY``/``CONTAINS``) so it never distorts local distances.
        """
        ...

    def shortest_path(
        self,
        src: NodeId,
        dst: NodeId,
        *,
        exclude_node_types: Collection[NodeType] = frozenset(),
        exclude_edge_types: Collection[EdgeType] = frozenset(),
    ) -> Path | None:
        """Return the shortest directed path, or ``None`` if unreachable."""
        ...

    def pagerank(self) -> dict[str, float]:
        """Return raw PageRank centrality scores over the full graph (sum ≈ 1.0 when non-empty)."""
        ...
