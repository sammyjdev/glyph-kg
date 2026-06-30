"""S-06..S-09: PathRetriever — shortest-path arm."""

from unittest.mock import MagicMock

from glyph.model.contract import ContextPack
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.path import PathRetriever
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


def _make_store() -> NetworkXStore:
    store = NetworkXStore()
    nodes = [
        Node(id="A", type=NodeType.ENTITY, label="Alpha"),
        Node(id="B", type=NodeType.ENTITY, label="Beta"),
        Node(id="C", type=NodeType.ENTITY, label="Gamma"),
    ]
    edges = [
        Edge(src="A", dst="B", type=EdgeType.REQUIRES),
        Edge(src="B", dst="C", type=EdgeType.RELATES_TO),
    ]
    store.upsert_nodes(nodes)
    store.upsert_edges(edges)
    return store


def _make_embedder(mapping: dict[str, list[float]]):
    """Stub embedder: returns a fixed vector per text."""
    emb = MagicMock()
    emb.embed.side_effect = lambda texts: [mapping[t] for t in texts]
    return emb


# --- S-06: happy path ---


def test_s06_path_found_returns_intermediate_nodes():
    # Query embedding closest to "Alpha" and "Gamma"; path A→B→C exists.
    store = _make_store()
    mapping = {
        "Alpha": [1.0, 0.0],
        "Beta": [0.0, 1.0],
        "Gamma": [0.9, 0.1],
        "find path from Alpha to Gamma": [1.0, 0.0],
    }
    embedder = _make_embedder(mapping)
    nodes = [
        Node(id=nid, type=NodeType.ENTITY, label=store._g.nodes[nid]["label"])
        for nid in store._g.nodes
    ]
    retriever = PathRetriever(store, embedder, nodes)

    pack = retriever.retrieve("find path from Alpha to Gamma", token_budget=2000)

    assert isinstance(pack, ContextPack)
    sources = {s.source for s in pack.segments}
    assert "A" in sources  # source anchor
    assert "C" in sources  # dest anchor
    assert "B" in sources  # intermediate


def test_s06_path_segments_include_edge_type():
    store = _make_store()
    mapping = {
        "Alpha": [1.0, 0.0],
        "Beta": [0.0, 1.0],
        "Gamma": [0.9, 0.1],
        "q": [1.0, 0.0],
    }
    embedder = _make_embedder(mapping)
    # Reconstruct nodes list from store
    nodes = [
        Node(id=nid, type=NodeType.ENTITY, label=store._g.nodes[nid]["label"])
        for nid in store._g.nodes
    ]
    retriever = PathRetriever(store, embedder, nodes)
    pack = retriever.retrieve("q", token_budget=2000)
    joined = " ".join(s.text for s in pack.segments)
    assert "requires" in joined or "relates_to" in joined


# --- S-07: no path fallback ---


def test_s07_no_path_fallback_returns_anchors():
    store = NetworkXStore()
    store.upsert_nodes(
        [
            Node(id="X", type=NodeType.ENTITY, label="Xray"),
            Node(id="Y", type=NodeType.ENTITY, label="Yankee"),
        ]
    )
    # No edges — no path between X and Y
    mapping = {"Xray": [1.0, 0.0], "Yankee": [0.0, 1.0], "q": [0.6, 0.4]}
    embedder = _make_embedder(mapping)
    nodes = [
        Node(id="X", type=NodeType.ENTITY, label="Xray"),
        Node(id="Y", type=NodeType.ENTITY, label="Yankee"),
    ]
    retriever = PathRetriever(store, embedder, nodes)
    pack = retriever.retrieve("q", token_budget=500)
    assert isinstance(pack, ContextPack)
    assert len(pack.segments) >= 1  # at least one anchor returned


# --- S-08: port conformance ---


def test_s08_port():
    store = NetworkXStore()
    nodes = [Node(id="A", type=NodeType.ENTITY, label="Alpha")]
    store.upsert_nodes(nodes)
    mapping = {"Alpha": [1.0, 0.0]}
    embedder = _make_embedder(mapping)
    retriever = PathRetriever(store, embedder, nodes)
    assert isinstance(retriever, Retriever)


# --- S-09: overlay excluded ---


def test_s09_community_overlay_excluded():
    store = NetworkXStore()
    store.upsert_nodes(
        [
            Node(id="A", type=NodeType.ENTITY, label="Alpha"),
            Node(id="C1", type=NodeType.COMMUNITY, label="Community1"),
            Node(id="B", type=NodeType.ENTITY, label="Beta"),
        ]
    )
    store.upsert_edges(
        [
            Edge(src="C1", dst="A", type=EdgeType.CONTAINS),
            Edge(src="C1", dst="B", type=EdgeType.CONTAINS),
        ]
    )
    # Only path from A to B goes through C1 (community overlay) — must return None/fallback
    mapping = {"Alpha": [1.0, 0.0], "Beta": [0.9, 0.1], "Community1": [0.5, 0.5], "q": [1.0, 0.0]}
    embedder = _make_embedder(mapping)
    nodes = [
        Node(id="A", type=NodeType.ENTITY, label="Alpha"),
        Node(id="B", type=NodeType.ENTITY, label="Beta"),
        Node(id="C1", type=NodeType.COMMUNITY, label="Community1"),
    ]
    retriever = PathRetriever(store, embedder, nodes)
    pack = retriever.retrieve("q", token_budget=500)
    # Should not contain C1 in sources (path through overlay is excluded)
    assert "C1" not in {s.source for s in pack.segments}
