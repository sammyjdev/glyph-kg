from collections.abc import Iterable
from pathlib import Path as FsPath

import pytest

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.store.networkx_store import NetworkXStore


def _node(nid: str) -> Node:
    return Node(id=nid, type=NodeType.ENTITY, label=nid)


def _ids(nodes: Iterable[Node]) -> set[str]:
    return {n.id for n in nodes}


def _edge_keys(edges: Iterable[Edge]) -> set[tuple[str, str, EdgeType]]:
    return {(e.src, e.dst, e.type) for e in edges}


@pytest.fixture
def store() -> NetworkXStore:
    s = NetworkXStore()
    # a -> b -> c, directed
    s.upsert_nodes([_node("a"), _node("b"), _node("c")])
    s.upsert_edges(
        [
            Edge(src="a", dst="b", type=EdgeType.RELATES_TO),
            Edge(src="b", dst="c", type=EdgeType.RELATES_TO),
        ]
    )
    return s


def test_neighbors_includes_anchor_and_direct_neighbors(store: NetworkXStore) -> None:
    sg = store.neighbors("a", hops=1)
    assert _ids(sg.nodes) == {"a", "b"}


def test_neighbors_expands_by_hops_ignoring_direction(store: NetworkXStore) -> None:
    assert _ids(store.neighbors("a", hops=2).nodes) == {"a", "b", "c"}
    # undirected expansion: c reaches a within 2 hops even though edges point forward
    assert _ids(store.neighbors("c", hops=2).nodes) == {"a", "b", "c"}


def test_neighbors_returns_induced_edges(store: NetworkXStore) -> None:
    sg = store.neighbors("a", hops=2)
    assert _edge_keys(sg.edges) == {
        ("a", "b", EdgeType.RELATES_TO),
        ("b", "c", EdgeType.RELATES_TO),
    }


def test_subgraph_unions_multiple_seeds(store: NetworkXStore) -> None:
    store.upsert_nodes([_node("x"), _node("y")])
    store.upsert_edges([Edge(src="x", dst="y", type=EdgeType.MENTIONS)])
    sg = store.subgraph(seed=["a", "x"], hops=1)
    assert _ids(sg.nodes) == {"a", "b", "x", "y"}


def test_shortest_path_is_directed(store: NetworkXStore) -> None:
    path = store.shortest_path("a", "c")
    assert path is not None
    assert path.nodes == ["a", "b", "c"]


def test_shortest_path_returns_none_against_edge_direction(store: NetworkXStore) -> None:
    assert store.shortest_path("c", "a") is None


def test_upsert_node_updates_existing_id(store: NetworkXStore) -> None:
    store.upsert_nodes([Node(id="a", type=NodeType.CONCEPT, label="renamed")])
    sg = store.neighbors("a", hops=0)
    (a,) = [n for n in sg.nodes if n.id == "a"]
    assert a.label == "renamed"
    assert a.type is NodeType.CONCEPT


def test_parallel_edge_types_between_same_pair_coexist() -> None:
    s = NetworkXStore()
    s.upsert_nodes([_node("a"), _node("b")])
    s.upsert_edges(
        [
            Edge(src="a", dst="b", type=EdgeType.RELATES_TO),
            Edge(src="a", dst="b", type=EdgeType.MENTIONS),
        ]
    )
    sg = s.neighbors("a", hops=1)
    assert _edge_keys(sg.edges) == {
        ("a", "b", EdgeType.RELATES_TO),
        ("a", "b", EdgeType.MENTIONS),
    }


def test_subgraph_skips_unknown_seed(store: NetworkXStore) -> None:
    assert _ids(store.subgraph(seed=["a", "ghost"], hops=1).nodes) == {"a", "b"}


def test_shortest_path_returns_none_for_unknown_node(store: NetworkXStore) -> None:
    assert store.shortest_path("a", "ghost") is None


@pytest.fixture
def hub_store() -> NetworkXStore:
    # structural path a -> b -> c -> d (a and d are 3 hops apart), plus a COMMUNITY
    # super-hub linked to a and d by CONTAINS (the overlay added by P7).
    s = NetworkXStore()
    s.upsert_nodes([_node("a"), _node("b"), _node("c"), _node("d")])
    s.upsert_nodes([Node(id="comm", type=NodeType.COMMUNITY, label="faction")])
    s.upsert_edges(
        [
            Edge(src="a", dst="b", type=EdgeType.RELATES_TO),
            Edge(src="b", dst="c", type=EdgeType.RELATES_TO),
            Edge(src="c", dst="d", type=EdgeType.RELATES_TO),
            Edge(src="comm", dst="a", type=EdgeType.CONTAINS),
            Edge(src="comm", dst="d", type=EdgeType.CONTAINS),
        ]
    )
    return s


def test_subgraph_without_exclusion_super_hub_collapses_distance(hub_store: NetworkXStore) -> None:
    # WITHOUT pruning: the CONTAINS super-hub teleports a..d to 2 hops (a->comm->d)
    # and leaks the COMMUNITY node. This documents the bug the exclusion fixes.
    got = _ids(hub_store.subgraph(seed=["a"], hops=2).nodes)
    assert "comm" in got
    assert "d" in got  # structurally 3 hops, but reachable via the hub


def test_subgraph_excludes_overlay_node_and_edge_types(hub_store: NetworkXStore) -> None:
    # WITH pruning: COMMUNITY/CONTAINS are invisible to local traversal, so distances
    # stay structural (d unreachable in 2 hops) and the hub never leaks.
    got = _ids(
        hub_store.subgraph(
            seed=["a"],
            hops=2,
            exclude_node_types={NodeType.COMMUNITY},
            exclude_edge_types={EdgeType.CONTAINS},
        ).nodes
    )
    assert got == {"a", "b", "c"}


def test_neighbors_forwards_exclusions(hub_store: NetworkXStore) -> None:
    got = _ids(
        hub_store.neighbors(
            "a",
            hops=1,
            exclude_node_types={NodeType.COMMUNITY},
            exclude_edge_types={EdgeType.CONTAINS},
        ).nodes
    )
    assert got == {"a", "b"}  # without exclusion this would also include "comm"


def test_shortest_path_excludes_edge_types(hub_store: NetworkXStore) -> None:
    assert hub_store.shortest_path("comm", "d").nodes == ["comm", "d"]
    # excluding CONTAINS removes the hub's only out-edges -> no path
    assert hub_store.shortest_path("comm", "d", exclude_edge_types={EdgeType.CONTAINS}) is None


def test_persistence_round_trip(store: NetworkXStore, tmp_path: FsPath) -> None:
    target = tmp_path / "graph.json"
    store.save(target)
    loaded = NetworkXStore.load(target)
    assert _ids(loaded.neighbors("a", hops=2).nodes) == {"a", "b", "c"}
    assert loaded.shortest_path("a", "c") is not None
    assert loaded.shortest_path("a", "c").nodes == ["a", "b", "c"]
