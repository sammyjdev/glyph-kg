"""NetworkX-backed GraphStore: the embedded, zero-server default backend."""

import json
from collections.abc import Collection, Sequence
from pathlib import Path as FsPath
from typing import Any

import networkx as nx

from glyph.model.edge import Edge, EdgeType
from glyph.model.graph import NodeId, Path, Subgraph
from glyph.model.node import Node, NodeType


class NetworkXStore:
    """A :class:`~glyph.store.port.GraphStore` over an in-memory ``MultiDiGraph``.

    Edges are keyed by ``(src, dst, type)`` so parallel relations between the
    same pair coexist. Neighborhood expansion is undirected (retrieval wants
    context regardless of arrow direction); ``shortest_path`` stays directed.
    """

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    # -- writes ---------------------------------------------------------------

    def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        for node in nodes:
            self._g.add_node(
                node.id, type=node.type.value, label=node.label, attrs=dict(node.attrs)
            )

    def upsert_edges(self, edges: Sequence[Edge]) -> None:
        for edge in edges:
            self._g.add_edge(
                edge.src,
                edge.dst,
                key=edge.type.value,
                type=edge.type.value,
                attrs=dict(edge.attrs),
            )

    # -- reads ----------------------------------------------------------------

    def neighbors(
        self,
        node: NodeId,
        hops: int,
        *,
        exclude_node_types: Collection[NodeType] = frozenset(),
        exclude_edge_types: Collection[EdgeType] = frozenset(),
    ) -> Subgraph:
        return self.subgraph(
            [node],
            hops,
            exclude_node_types=exclude_node_types,
            exclude_edge_types=exclude_edge_types,
        )

    def subgraph(
        self,
        seed: Sequence[NodeId],
        hops: int,
        *,
        exclude_node_types: Collection[NodeType] = frozenset(),
        exclude_edge_types: Collection[EdgeType] = frozenset(),
    ) -> Subgraph:
        g = self._projection(exclude_node_types, exclude_edge_types)
        undirected = g.to_undirected(as_view=True)
        reachable: set[str] = set()
        for src in seed:
            if src not in g:
                continue
            reachable.update(nx.single_source_shortest_path_length(undirected, src, cutoff=hops))
        nodes = [self._to_node(nid) for nid in reachable]
        edges = [
            self._to_edge(u, v, key, data)
            for u, v, key, data in g.edges(keys=True, data=True)
            if u in reachable and v in reachable
        ]
        return Subgraph(nodes=nodes, edges=edges)

    def shortest_path(
        self,
        src: NodeId,
        dst: NodeId,
        *,
        exclude_node_types: Collection[NodeType] = frozenset(),
        exclude_edge_types: Collection[EdgeType] = frozenset(),
    ) -> Path | None:
        g = self._projection(exclude_node_types, exclude_edge_types)
        if src not in g or dst not in g:
            return None
        try:
            ids = nx.shortest_path(g, src, dst)
        except nx.NetworkXNoPath:
            return None
        return Path(nodes=list(ids))

    # -- persistence ----------------------------------------------------------

    def save(self, path: FsPath) -> None:
        payload = {
            "nodes": [self._to_node(nid).model_dump() for nid in self._g.nodes],
            "edges": [
                self._to_edge(u, v, key, data).model_dump()
                for u, v, key, data in self._g.edges(keys=True, data=True)
            ],
        }
        FsPath(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: FsPath) -> "NetworkXStore":
        payload = json.loads(FsPath(path).read_text(encoding="utf-8"))
        store = cls()
        store.upsert_nodes([Node.model_validate(n) for n in payload["nodes"]])
        store.upsert_edges([Edge.model_validate(e) for e in payload["edges"]])
        return store

    # -- internals ------------------------------------------------------------

    def _projection(
        self,
        exclude_node_types: Collection[NodeType],
        exclude_edge_types: Collection[EdgeType],
    ) -> Any:
        """A read-only view with the excluded node/edge types pruned from traversal.

        Returns the graph itself when nothing is excluded (the common, cheap path).
        Otherwise a ``subgraph_view`` so an overlay (e.g. COMMUNITY super-hubs) can be
        made invisible without mutating or copying the stored graph.
        """
        if not exclude_node_types and not exclude_edge_types:
            return self._g
        excl_nodes = {t.value for t in exclude_node_types}
        excl_edges = {t.value for t in exclude_edge_types}

        def keep_node(nid: str) -> bool:
            return self._g.nodes[nid]["type"] not in excl_nodes

        def keep_edge(u: str, v: str, key: str) -> bool:
            return key not in excl_edges

        return nx.subgraph_view(self._g, filter_node=keep_node, filter_edge=keep_edge)

    def _to_node(self, nid: str) -> Node:
        data = self._g.nodes[nid]
        return Node(
            id=nid, type=NodeType(data["type"]), label=data["label"], attrs=dict(data["attrs"])
        )

    def _to_edge(self, src: str, dst: str, key: str, data: dict[str, Any]) -> Edge:
        return Edge(src=src, dst=dst, type=EdgeType(data["type"]), attrs=dict(data["attrs"]))
