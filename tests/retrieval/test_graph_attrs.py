from collections.abc import Sequence

from glyph.embed.port import Vector
from glyph.model.contract import count_tokens
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.graph import GraphRetriever
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
        Node(
            id="goblin",
            type=NodeType.ENTITY,
            label="Goblin",
            attrs={"challenge_rating": "5"},
        ),
        Node(id="fogo", type=NodeType.CONCEPT, label="fogo"),
        Node(id="caverna", type=NodeType.CONCEPT, label="caverna"),
    ]


def _store(nodes: list[Node]) -> NetworkXStore:
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([Edge(src="goblin", dst="fogo", type=EdgeType.RESISTS)])
    return store


def test_default_omits_attrs_unchanged() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes, hops=1)
    result = retriever.retrieve("goblin", token_budget=1000)
    goblin = next(s for s in result.segments if s.source == "goblin")
    assert goblin.text == "Goblin — resists fogo"


def test_include_attrs_appends_attr_values() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(
        store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes, hops=1, include_attrs=True
    )
    result = retriever.retrieve("goblin", token_budget=1000)
    goblin = next(s for s in result.segments if s.source == "goblin")
    assert goblin.text == "Goblin — resists fogo — challenge_rating: 5"
    assert "challenge_rating" in goblin.text
    assert "5" in goblin.text


def test_include_attrs_with_no_relations() -> None:
    nodes = [
        Node(
            id="caverna",
            type=NodeType.CONCEPT,
            label="caverna",
            attrs={"tamanho": "grande"},
        ),
        Node(id="fogo", type=NodeType.CONCEPT, label="fogo"),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    retriever = GraphRetriever(
        store=store, embedder=_FakeEmbedder(), nodes=nodes, hops=1, include_attrs=True
    )
    result = retriever.retrieve("caverna", token_budget=1000)
    caverna = next(s for s in result.segments if s.source == "caverna")
    assert caverna.text == "caverna — tamanho: grande"


def test_attrs_reduce_segments_under_tight_budget() -> None:
    long_value = "x" * 200
    nodes = [
        Node(id="goblin", type=NodeType.ENTITY, label="Goblin"),
        Node(id="fogo", type=NodeType.CONCEPT, label="fogo", attrs={"alignment": long_value}),
        Node(id="caverna", type=NodeType.CONCEPT, label="caverna", attrs={"alignment": long_value}),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="goblin", dst="fogo", type=EdgeType.RESISTS),
            Edge(src="goblin", dst="caverna", type=EdgeType.INHABITS),
        ]
    )
    retriever_without = GraphRetriever(
        store=store, embedder=_FakeEmbedder(), nodes=nodes, hops=1, anchors=1
    )
    retriever_with = GraphRetriever(
        store=store, embedder=_FakeEmbedder(), nodes=nodes, hops=1, anchors=1, include_attrs=True
    )
    pack_without = retriever_without.retrieve("goblin", token_budget=30)
    pack_with = retriever_with.retrieve("goblin", token_budget=30)
    assert len(pack_with.segments) < len(pack_without.segments)
    assert count_tokens(pack_with.segments[0].text) > 0
