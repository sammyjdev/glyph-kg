"""P7 community detection + graph elements (deterministic, structural-only)."""

from collections.abc import Iterable

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.community import Community, detect_communities, to_graph_elements
from glyph.store.networkx_store import NetworkXStore


def _fn(nid: str) -> Node:
    return Node(id=nid, type=NodeType.FUNCTION, label=nid)


def _members(comms: Iterable[Community]) -> list[list[str]]:
    return sorted(sorted(c.members) for c in comms)


def _two_triangles() -> tuple[NetworkXStore, list[Node]]:
    # two dense triangles {a,b,c} and {d,e,f} joined by a single bridge c-d
    nodes = [_fn(x) for x in "abcdef"]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    pairs = [("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f"), ("c", "d")]
    store.upsert_edges([Edge(src=u, dst=v, type=EdgeType.CALLS) for u, v in pairs])
    return store, nodes


def test_detect_partitions_two_clusters() -> None:
    store, nodes = _two_triangles()
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a", "b", "c"], ["d", "e", "f"]]


def test_detect_ignores_preexisting_community_nodes() -> None:
    store, nodes = _two_triangles()
    # a stale overlay from a previous build must not be treated as a member
    store.upsert_nodes([Node(id="comm", type=NodeType.COMMUNITY, label="old")])
    store.upsert_edges([Edge(src="comm", dst="a", type=EdgeType.CONTAINS)])
    nodes = [*nodes, Node(id="comm", type=NodeType.COMMUNITY, label="old")]
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a", "b", "c"], ["d", "e", "f"]]


def test_detect_clusters_on_structural_edges_only() -> None:
    # nodes joined only by a document edge (MENTIONS) have no structural links,
    # so each is its own community — ADRs/entities never merge with code.
    nodes = [_fn("a"), _fn("b"), _fn("c")]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="a", dst="b", type=EdgeType.MENTIONS),
            Edge(src="b", dst="c", type=EdgeType.MENTIONS),
        ]
    )
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a"], ["b"], ["c"]]


def test_detect_is_reproducible_with_seed() -> None:
    store, nodes = _two_triangles()
    run1 = [c.id for c in detect_communities(store, nodes, seed=0)]
    run2 = [c.id for c in detect_communities(store, nodes, seed=0)]
    assert run1 == run2
    assert all(cid.startswith("community:") for cid in run1)


def test_detect_ignores_self_loops() -> None:
    # a recursive call (CALLS itself) is a self-loop in the graph; it must be
    # skipped, not break detection.
    nodes = [_fn("a"), _fn("b"), _fn("c")]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="a", dst="a", type=EdgeType.CALLS),  # recursion
            Edge(src="a", dst="b", type=EdgeType.CALLS),
            Edge(src="b", dst="c", type=EdgeType.CALLS),
            Edge(src="a", dst="c", type=EdgeType.CALLS),
        ]
    )
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a", "b", "c"]]


def test_to_graph_elements_builds_community_nodes_and_contains_edges() -> None:
    comms = [Community(members=("a", "b", "c"))]
    comm_nodes, edges = to_graph_elements(comms)
    (node,) = comm_nodes
    assert node.type is NodeType.COMMUNITY
    assert node.id == comms[0].id
    assert node.attrs["members"] == 3
    assert {(e.src, e.dst, e.type) for e in edges} == {
        (node.id, "a", EdgeType.CONTAINS),
        (node.id, "b", EdgeType.CONTAINS),
        (node.id, "c", EdgeType.CONTAINS),
    }
