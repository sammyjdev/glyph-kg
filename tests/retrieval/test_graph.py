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


def test_retrieve_excludes_community_overlay_from_local_results() -> None:
    # The store carries a COMMUNITY super-hub (CONTAINS goblin + a distant member).
    # Local retrieval must not surface the overlay node nor teleport through CONTAINS.
    nodes = _nodes()
    store = _store(nodes)
    store.upsert_nodes([Node(id="comm", type=NodeType.COMMUNITY, label="faction")])
    store.upsert_edges(
        [
            Edge(src="comm", dst="goblin", type=EdgeType.CONTAINS),
            Edge(src="comm", dst="caverna", type=EdgeType.CONTAINS),
        ]
    )
    # anchors=1 isolates traversal from anchor selection: only goblin anchors, so
    # caverna can appear ONLY if pulled through the CONTAINS hub (it must not).
    retriever = GraphRetriever(
        store=store, embedder=_FakeEmbedder(), nodes=nodes, hops=1, anchors=1
    )
    sources = {s.source for s in retriever.retrieve("goblin", token_budget=1000).segments}
    assert "comm" not in sources  # overlay node never leaks into local context
    assert "caverna" not in sources  # not pulled in via the CONTAINS hub
    assert sources == {"goblin", "fogo"}


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
    # Neighbors score below 1.0 (anchors); ties within non-anchors break by source.
    non_anchor = [s.source for s in result.segments if s.score < 1.0]
    assert non_anchor == sorted(non_anchor)


class _ScoredEmbedder:
    """Embedder where 'fogo' partially aligns with 'Goblin' query; 'caverna' does not."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        m: dict[str, list[float]] = {
            "Goblin": [0.0, 0.0, 1.0],
            "fogo": [0.0, 1.0, 1.0],  # shares [0,0,1] component with Goblin
            "caverna": [1.0, 0.0, 0.0],  # orthogonal to Goblin
        }
        return [m.get(t, [0.0, 0.0, 0.0]) for t in texts]


def test_neighbor_score_reflects_query_relevance() -> None:
    """Neighbor semantically closer to the query gets a higher score than an unrelated one."""
    nodes = [
        Node(id="goblin", type=NodeType.ENTITY, label="Goblin"),
        Node(id="fogo", type=NodeType.CONCEPT, label="fogo"),
        Node(id="caverna", type=NodeType.CONCEPT, label="caverna"),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="goblin", dst="fogo", type=EdgeType.RESISTS),
            Edge(src="goblin", dst="caverna", type=EdgeType.INHABITS),
        ]
    )
    retriever = GraphRetriever(
        store=store, embedder=_ScoredEmbedder(), nodes=nodes, hops=1, anchors=1
    )
    result = retriever.retrieve("Goblin", token_budget=1000)
    scores = {s.source: s.score for s in result.segments}
    assert scores["fogo"] > scores["caverna"]
