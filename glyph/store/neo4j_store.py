"""Neo4j-backed GraphStore adapter.

Requires the ``neo4j`` extra:  pip install "glyph-kg[neo4j]"

Run the contract tests (needs Docker):  pytest -m neo4j
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence

from neo4j import GraphDatabase

from glyph.model.edge import Edge, EdgeType
from glyph.model.graph import NodeId, Path, Subgraph
from glyph.model.node import Node, NodeType


class Neo4jStore:
    """A :class:`~glyph.store.port.GraphStore` backed by Neo4j.

    All nodes carry the Cypher label ``:Node`` and are keyed by the ``id``
    property (unique constraint applied on construction).  Edges use the
    EdgeType value (upper-cased) as the Cypher relationship type so parallel
    edge types between the same pair coexist naturally.

    Semantics match :class:`~glyph.store.networkx_store.NetworkXStore` exactly:
    - Neighborhood expansion is **undirected**; exclusions prune before traversal.
    - ``shortest_path`` is **directed**; respects both node and edge exclusions.
    """

    def __init__(self, uri: str, *, auth: tuple[str, str]) -> None:
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._ensure_constraint()

    def _ensure_constraint(self) -> None:
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")

    def close(self) -> None:
        self._driver.close()

    # -- writes ---------------------------------------------------------------

    def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        if not nodes:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Node {id: row.id})
                SET n.type = row.type,
                    n.label = row.label,
                    n.attrs = row.attrs
                """,
                rows=[
                    {
                        "id": node.id,
                        "type": node.type.value,
                        "label": node.label,
                        "attrs": json.dumps(dict(node.attrs)),
                    }
                    for node in nodes
                ],
            )

    def upsert_edges(self, edges: Sequence[Edge]) -> None:
        if not edges:
            return
        # Group by edge type so we can build a type-specific Cypher query per group.
        # This gives us (src, dst, type) as the natural MERGE key.
        by_type: dict[str, list[dict[str, str]]] = {}
        for edge in edges:
            rel_type = edge.type.value.upper()
            by_type.setdefault(rel_type, []).append(
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "attrs": json.dumps(dict(edge.attrs)),
                }
            )
        with self._driver.session() as session:
            for rel_type, rows in by_type.items():
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:Node {{id: row.src}})
                    MATCH (b:Node {{id: row.dst}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.attrs = row.attrs
                    """,
                    rows=rows,
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
        ex_nodes = [t.value for t in exclude_node_types]
        ex_edges = [t.value.upper() for t in exclude_edge_types]

        # Collect reachable node IDs per seed via undirected BFS, then union.
        reachable: set[str] = set()
        with self._driver.session() as session:
            for seed_id in seed:
                # Check seed exists and is not excluded.
                check = session.run(
                    "MATCH (s:Node {id: $id}) WHERE NOT s.type IN $ex_nodes RETURN s.id",
                    id=seed_id,
                    ex_nodes=ex_nodes,
                )
                if not check.single():
                    continue  # unknown or excluded seed — skip

                # Variable-length undirected traversal with per-rel type filtering.
                # ALL(rel IN r WHERE ...) enforces exclusions on every hop in the path.
                result = session.run(
                    f"""
                    MATCH (s:Node {{id: $id}})
                    MATCH p = (s)-[r*0..{hops}]-(n:Node)
                    WHERE ALL(rel IN r WHERE NOT type(rel) IN $ex_edges)
                      AND NOT n.type IN $ex_nodes
                    RETURN collect(DISTINCT n.id) AS ids
                    """,
                    id=seed_id,
                    ex_nodes=ex_nodes,
                    ex_edges=ex_edges,
                )
                record = result.single()
                if record:
                    reachable.update(record["ids"])
                # Always include the anchor itself (matched by [*0..N] with n=s).

        if not reachable:
            return Subgraph(nodes=[], edges=[])

        return self._induce(list(reachable), ex_edges)

    def _induce(self, node_ids: list[str], ex_edges: list[str]) -> Subgraph:
        """Return Node/Edge objects for the induced directed subgraph on node_ids."""
        with self._driver.session() as session:
            node_result = session.run(
                "UNWIND $ids AS nid MATCH (n:Node {id: nid}) RETURN n",
                ids=node_ids,
            )
            nodes = [_to_node(record["n"]) for record in node_result]

            edge_result = session.run(
                """
                UNWIND $ids AS aid
                MATCH (a:Node {id: aid})-[r]->(b:Node)
                WHERE b.id IN $ids
                  AND NOT type(r) IN $ex_edges
                RETURN a.id AS src, b.id AS dst, type(r) AS rel_type, r.attrs AS attrs
                """,
                ids=node_ids,
                ex_edges=ex_edges,
            )
            edges = [_to_edge(record) for record in edge_result]

        return Subgraph(nodes=nodes, edges=edges)

    def shortest_path(
        self,
        src: NodeId,
        dst: NodeId,
        *,
        exclude_node_types: Collection[NodeType] = frozenset(),
        exclude_edge_types: Collection[EdgeType] = frozenset(),
    ) -> Path | None:
        ex_nodes = [t.value for t in exclude_node_types]
        ex_edges = [t.value.upper() for t in exclude_edge_types]

        with self._driver.session() as session:
            # Verify both endpoints exist in the projection.
            exist = session.run(
                """
                MATCH (s:Node {id: $src}) WHERE NOT s.type IN $ex_nodes
                MATCH (d:Node {id: $dst}) WHERE NOT d.type IN $ex_nodes
                RETURN 1
                """,
                src=src,
                dst=dst,
                ex_nodes=ex_nodes,
            )
            if not exist.single():
                return None

            # shortestPath with post-filter for excluded types.
            # NONE() predicates on relationships and interior nodes in the path.
            result = session.run(
                """
                MATCH p = shortestPath(
                    (s:Node {id: $src})-[*]->(d:Node {id: $dst})
                )
                WHERE NONE(r IN relationships(p) WHERE type(r) IN $ex_edges)
                  AND NONE(n IN nodes(p) WHERE n.type IN $ex_nodes)
                RETURN [n IN nodes(p) | n.id] AS ids
                """,
                src=src,
                dst=dst,
                ex_nodes=ex_nodes,
                ex_edges=ex_edges,
            )
            record = result.single()
            if not record:
                return None
            return Path(nodes=list(record["ids"]))

    def pagerank(self) -> dict[str, float]:
        """Return raw PageRank scores (sum ≈ 1.0). Computed client-side via networkx —
        avoids requiring the Neo4j GDS plugin for a feature this small.

        Computed on the undirected projection to match
        :meth:`~glyph.store.networkx_store.NetworkXStore.pagerank` exactly: PageRank
        on the raw directed graph ranks sink nodes above hubs, which inverts the
        centrality this method is meant to capture.
        """
        import networkx as nx

        with self._driver.session() as session:
            node_ids = [record["id"] for record in session.run("MATCH (n:Node) RETURN n.id AS id")]
            if not node_ids:
                return {}
            edge_result = session.run(
                "MATCH (a:Node)-[r]->(b:Node) RETURN a.id AS src, b.id AS dst"
            )
            g = nx.DiGraph()
            g.add_nodes_from(node_ids)
            g.add_edges_from((record["src"], record["dst"]) for record in edge_result)
        result: dict[str, float] = nx.pagerank(g.to_undirected())
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_node(record: object) -> Node:
    n = record
    return Node(
        id=n["id"],  # type: ignore[index]
        type=NodeType(n["type"]),  # type: ignore[index]
        label=n["label"],  # type: ignore[index]
        attrs=json.loads(n["attrs"] or "{}"),  # type: ignore[index]
    )


def _to_edge(record: object) -> Edge:
    r = record
    return Edge(
        src=r["src"],  # type: ignore[index]
        dst=r["dst"],  # type: ignore[index]
        type=EdgeType(r["rel_type"].lower()),  # type: ignore[index]
        attrs=json.loads(r["attrs"] or "{}"),  # type: ignore[index]
    )
