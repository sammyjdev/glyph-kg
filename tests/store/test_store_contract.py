"""Backend-agnostic contract tests for the GraphStore protocol.

All tests run against NetworkXStore (always) and Neo4jStore (when ``-m neo4j``
is passed; requires Docker).  The ``store`` and ``hub_store`` fixtures are
parametrized in ``conftest.py``.
"""

from collections.abc import Iterable

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.store.port import GraphStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids(nodes: Iterable[Node]) -> set[str]:
    return {n.id for n in nodes}


def _edge_keys(edges: Iterable[Edge]) -> set[tuple[str, str, EdgeType]]:
    return {(e.src, e.dst, e.type) for e in edges}


# ---------------------------------------------------------------------------
# Linear chain tests (use ``store`` fixture: a->b->c pre-populated)
# ---------------------------------------------------------------------------


def test_neighbors_includes_anchor_and_direct_neighbors(store: GraphStore) -> None:
    sg = store.neighbors("a", hops=1)
    assert _ids(sg.nodes) == {"a", "b"}


def test_neighbors_expands_by_hops_ignoring_direction(store: GraphStore) -> None:
    assert _ids(store.neighbors("a", hops=2).nodes) == {"a", "b", "c"}
    # undirected expansion: c reaches a within 2 hops even though edges point forward
    assert _ids(store.neighbors("c", hops=2).nodes) == {"a", "b", "c"}


def test_neighbors_returns_induced_edges(store: GraphStore) -> None:
    sg = store.neighbors("a", hops=2)
    assert _edge_keys(sg.edges) == {
        ("a", "b", EdgeType.RELATES_TO),
        ("b", "c", EdgeType.RELATES_TO),
    }


def test_subgraph_unions_multiple_seeds(store: GraphStore) -> None:
    store.upsert_nodes(
        [
            Node(id="x", type=NodeType.ENTITY, label="x"),
            Node(id="y", type=NodeType.ENTITY, label="y"),
        ]
    )
    store.upsert_edges([Edge(src="x", dst="y", type=EdgeType.MENTIONS)])
    sg = store.subgraph(seed=["a", "x"], hops=1)
    assert _ids(sg.nodes) == {"a", "b", "x", "y"}


def test_shortest_path_is_directed(store: GraphStore) -> None:
    path = store.shortest_path("a", "c")
    assert path is not None
    assert path.nodes == ["a", "b", "c"]


def test_shortest_path_returns_none_against_edge_direction(store: GraphStore) -> None:
    assert store.shortest_path("c", "a") is None


def test_upsert_node_updates_existing_id(store: GraphStore) -> None:
    store.upsert_nodes([Node(id="a", type=NodeType.CONCEPT, label="renamed")])
    sg = store.neighbors("a", hops=0)
    (a,) = [n for n in sg.nodes if n.id == "a"]
    assert a.label == "renamed"
    assert a.type is NodeType.CONCEPT


def test_parallel_edge_types_between_same_pair_coexist(store: GraphStore) -> None:
    # Add a second edge type between a and b (RELATES_TO already exists from fixture)
    store.upsert_edges([Edge(src="a", dst="b", type=EdgeType.MENTIONS)])
    sg = store.neighbors("a", hops=1)
    keys = _edge_keys(sg.edges)
    assert ("a", "b", EdgeType.RELATES_TO) in keys
    assert ("a", "b", EdgeType.MENTIONS) in keys


def test_subgraph_skips_unknown_seed(store: GraphStore) -> None:
    assert _ids(store.subgraph(seed=["a", "ghost"], hops=1).nodes) == {"a", "b"}


def test_shortest_path_returns_none_for_unknown_node(store: GraphStore) -> None:
    assert store.shortest_path("a", "ghost") is None


# ---------------------------------------------------------------------------
# Hub topology tests (use ``hub_store`` fixture: a->b->c->d + COMMUNITY/CONTAINS)
# ---------------------------------------------------------------------------


def test_subgraph_without_exclusion_super_hub_collapses_distance(
    hub_store: GraphStore,
) -> None:
    # WITHOUT pruning: the CONTAINS super-hub teleports a..d to 2 hops (a<-comm->d)
    # and leaks the COMMUNITY node.
    got = _ids(hub_store.subgraph(seed=["a"], hops=2).nodes)
    assert "comm" in got
    assert "d" in got  # structurally 3 hops, but reachable via the hub


def test_subgraph_excludes_overlay_node_and_edge_types(hub_store: GraphStore) -> None:
    # WITH pruning: COMMUNITY/CONTAINS are invisible to local traversal.
    got = _ids(
        hub_store.subgraph(
            seed=["a"],
            hops=2,
            exclude_node_types={NodeType.COMMUNITY},
            exclude_edge_types={EdgeType.CONTAINS},
        ).nodes
    )
    assert got == {"a", "b", "c"}


def test_neighbors_forwards_exclusions(hub_store: GraphStore) -> None:
    got = _ids(
        hub_store.neighbors(
            "a",
            hops=1,
            exclude_node_types={NodeType.COMMUNITY},
            exclude_edge_types={EdgeType.CONTAINS},
        ).nodes
    )
    assert got == {"a", "b"}  # without exclusion this would also include "comm"


def test_shortest_path_excludes_edge_types(hub_store: GraphStore) -> None:
    assert hub_store.shortest_path("comm", "d") is not None
    assert hub_store.shortest_path("comm", "d").nodes == ["comm", "d"]
    # excluding CONTAINS removes the hub's only out-edges -> no path
    assert hub_store.shortest_path("comm", "d", exclude_edge_types={EdgeType.CONTAINS}) is None
