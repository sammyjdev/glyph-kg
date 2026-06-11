from collections.abc import Sequence

from glyph.embed.port import Vector
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


class _FakeEmbedder:
    """Deterministic 3-dim keyword embedder: [fogo, caverna, goblin]."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> Vector:
        low = text.lower()
        return [
            1.0 if "fogo" in low else 0.0,
            1.0 if "caverna" in low else 0.0,
            1.0 if "goblin" in low else 0.0,
        ]


def _nodes() -> list[Node]:
    return [
        Node(id="goblin", type=NodeType.ENTITY, label="Goblin"),
        Node(id="fogo", type=NodeType.CONCEPT, label="fogo"),
        Node(id="caverna", type=NodeType.CONCEPT, label="caverna"),
    ]


def _store(nodes: list[Node]) -> NetworkXStore:
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([Edge(src="goblin", dst="fogo", type=EdgeType.RESISTS)])
    return store


def test_graph_retriever_satisfies_the_port() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes)
    assert isinstance(retriever, Retriever)


def test_retrieve_anchors_query_and_returns_neighborhood() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes, hops=1)
    result = retriever.retrieve("fogo", token_budget=1000)
    assert result.mode == "graph"
    sources = {s.source for s in result.segments}
    assert "fogo" in sources  # anchored node
    assert "goblin" in sources  # its neighbor (resists fogo)


def test_segment_text_includes_relations() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes, hops=1)
    result = retriever.retrieve("goblin", token_budget=1000)
    goblin = next(s for s in result.segments if s.source == "goblin")
    assert "resists" in goblin.text
    assert "fogo" in goblin.text


def test_non_anchor_neighbors_break_score_ties_by_source() -> None:
    nodes = [
        Node(id="goblin", type=NodeType.ENTITY, label="Goblin"),
        Node(id="fogo", type=NodeType.CONCEPT, label="fogo"),
        Node(id="brasa", type=NodeType.CONCEPT, label="brasa"),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="goblin", dst="fogo", type=EdgeType.RESISTS),
            Edge(src="goblin", dst="brasa", type=EdgeType.IMMUNE_TO),
        ]
    )
    retriever = GraphRetriever(store=store, embedder=_FakeEmbedder(), nodes=nodes, hops=1)
    result = retriever.retrieve("goblin", token_budget=1000)
    tied = [s.source for s in result.segments if s.score == 0.5]
    assert tied == sorted(tied)  # deterministic order among tied neighbors
