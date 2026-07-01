from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.graph import GraphRetriever
from glyph.store.networkx_store import NetworkXStore


def _hub_graph() -> tuple[NetworkXStore, list[Node]]:
    """Three nodes: hub connected to both leaves; leaves connected to nothing else."""
    nodes = [
        Node(id="hub", type=NodeType.ENTITY, label="Hub"),
        Node(id="leaf_a", type=NodeType.CONCEPT, label="leaf_a"),
        Node(id="leaf_b", type=NodeType.CONCEPT, label="leaf_b"),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([
        Edge(src="hub", dst="leaf_a", type=EdgeType.RESISTS),
        Edge(src="hub", dst="leaf_b", type=EdgeType.RESISTS),
    ])
    return store, nodes


def test_s04_pagerank_sums_to_one() -> None:
    """pagerank() scores must sum to approximately 1.0."""
    store, _ = _hub_graph()
    pr = store.pagerank()
    assert abs(sum(pr.values()) - 1.0) < 1e-6


def test_s04_pagerank_hub_is_highest() -> None:
    """The hub node (most edges) must have the highest raw pagerank score."""
    store, _ = _hub_graph()
    pr = store.pagerank()
    assert pr["hub"] > pr["leaf_a"]
    assert pr["hub"] > pr["leaf_b"]


def _anchored_hub_graph() -> tuple[NetworkXStore, list[Node]]:
    """anchor -> hub -> {leaf_a, leaf_b}. `anchor` is first, so it wins cosine ties.

    hub's undirected degree is 3 (anchor, leaf_a, leaf_b) vs 1 for each leaf, giving a
    real, non-trivial PageRank difference among non-anchor nodes.
    """
    nodes = [
        Node(id="anchor", type=NodeType.ENTITY, label="Anchor"),
        Node(id="hub", type=NodeType.ENTITY, label="Hub"),
        Node(id="leaf_a", type=NodeType.CONCEPT, label="leaf_a"),
        Node(id="leaf_b", type=NodeType.CONCEPT, label="leaf_b"),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([
        Edge(src="anchor", dst="hub", type=EdgeType.RESISTS),
        Edge(src="hub", dst="leaf_a", type=EdgeType.RESISTS),
        Edge(src="hub", dst="leaf_b", type=EdgeType.RESISTS),
    ])
    return store, nodes


class _EqualEmbedder:
    """All nodes embed to the same vector — cosine tie — so PageRank breaks it."""
    def embed(self, texts):
        return [[1.0, 0.0, 0.0]] * len(texts)


def test_s05_pagerank_weight_raises_hub_score() -> None:
    """With pagerank_weight=0.5, a non-anchor hub outscores same-cosine leaves.

    `anchor` (first in `nodes`, wins the cosine tie) is the anchor, so its own
    score is hard-coded to 1.0 and doesn't exercise the blend. `hub` is a
    non-anchor node whose cosine ties with the leaves but whose higher degree
    (3 vs 1) gives it a higher PageRank — the blend must surface that.
    """
    store, nodes = _anchored_hub_graph()

    retriever = GraphRetriever(
        store=store,
        embedder=_EqualEmbedder(),
        nodes=nodes,
        hops=2,
        anchors=1,
        pagerank_weight=0.5,
    )
    result = retriever.retrieve("anything", token_budget=10_000)
    scores = {s.source: s.score for s in result.segments}
    assert scores["hub"] > scores["leaf_a"]
    assert scores["hub"] > scores["leaf_b"]


def test_s05_pagerank_weight_zero_keeps_pure_cosine_tie() -> None:
    """With pagerank_weight=0.0, hub and leaves stay tied (pure cosine, no PageRank)."""
    store, nodes = _anchored_hub_graph()

    retriever = GraphRetriever(
        store=store,
        embedder=_EqualEmbedder(),
        nodes=nodes,
        hops=2,
        anchors=1,
        pagerank_weight=0.0,
    )
    result = retriever.retrieve("anything", token_budget=10_000)
    scores = {s.source: s.score for s in result.segments}
    assert scores["hub"] == scores["leaf_a"] == scores["leaf_b"]
