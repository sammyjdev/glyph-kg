"""S-10..S-13: MultiAnchorRetriever."""
import pytest
from unittest.mock import MagicMock
from glyph.model.contract import ContextPack
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.multi_anchor import MultiAnchorRetriever, _extract_named_entities
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


def _embedder(mapping: dict[str, list[float]]):
    emb = MagicMock()
    emb.embed.side_effect = lambda texts: [mapping[t] for t in texts]
    return emb


# --- S-10: entity extraction ---

def test_s10_camelcase_extracted():
    assert "AuthService" in _extract_named_entities("how does AuthService connect to PostgreSQL?")
    assert "PostgreSQL" in _extract_named_entities("how does AuthService connect to PostgreSQL?")


def test_s10_dec_id_extracted():
    assert "dec-121" in _extract_named_entities("what does dec-121 require?")


def test_s10_adr_id_extracted():
    assert "ADR-G7" in _extract_named_entities("show me ADR-G7 relations")


def test_s10_plain_query_returns_empty():
    # No named entities in an all-lowercase query
    result = _extract_named_entities("what is the main purpose of this system?")
    assert result == []


# --- S-11: multi-seed subgraph ---

def test_s11_multi_seed_finds_connection():
    """Query names two entities that are connected; the connection node appears in results."""
    store = NetworkXStore()
    nodes = [
        Node(id="A", type=NodeType.ENTITY, label="AuthService"),
        Node(id="B", type=NodeType.ENTITY, label="Connector"),
        Node(id="C", type=NodeType.ENTITY, label="PostgreSQL"),
    ]
    edges = [
        Edge(src="A", dst="B", type=EdgeType.REQUIRES),
        Edge(src="B", dst="C", type=EdgeType.REQUIRES),
    ]
    store.upsert_nodes(nodes)
    store.upsert_edges(edges)

    mapping = {
        "AuthService": [1.0, 0.0, 0.0],
        "Connector":   [0.0, 1.0, 0.0],
        "PostgreSQL":  [0.0, 0.0, 1.0],
        "AuthService": [1.0, 0.0, 0.0],  # entity embed
        "PostgreSQL":  [0.0, 0.0, 1.0],  # entity embed
        "how does AuthService connect to PostgreSQL?": [0.5, 0.0, 0.5],
    }
    # Deduplicate mapping keys (Python keeps last)
    m = {
        "AuthService": [1.0, 0.0, 0.0],
        "Connector":   [0.0, 1.0, 0.0],
        "PostgreSQL":  [0.0, 0.0, 1.0],
        "how does AuthService connect to PostgreSQL?": [0.5, 0.0, 0.5],
    }
    embedder = _embedder(m)
    retriever = MultiAnchorRetriever(store, embedder, nodes)
    pack = retriever.retrieve("how does AuthService connect to PostgreSQL?", token_budget=2000)
    sources = {s.source for s in pack.segments}
    # Connector (B) sits between A and C and should appear in the expanded subgraph
    assert "B" in sources


# --- S-12: fallback when no named entities ---

def test_s12_fallback_uses_embedding_anchors():
    store = NetworkXStore()
    nodes = [
        Node(id="X", type=NodeType.ENTITY, label="alpha"),
        Node(id="Y", type=NodeType.ENTITY, label="beta"),
    ]
    store.upsert_nodes(nodes)
    m = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0],
         "what is the system purpose?": [0.8, 0.2]}
    embedder = _embedder(m)
    retriever = MultiAnchorRetriever(store, embedder, nodes)
    pack = retriever.retrieve("what is the system purpose?", token_budget=500)
    assert isinstance(pack, ContextPack)
    assert len(pack.segments) >= 1


# --- S-13: port conformance ---

def test_s13_port():
    store = NetworkXStore()
    nodes = [Node(id="A", type=NodeType.ENTITY, label="alpha")]
    store.upsert_nodes(nodes)
    m = {"alpha": [1.0, 0.0]}
    embedder = _embedder(m)
    assert isinstance(MultiAnchorRetriever(store, embedder, nodes), Retriever)
