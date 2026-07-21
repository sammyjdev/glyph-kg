from glyph.model.contract import Segment, count_tokens, estimate_tokens, pack


def test_count_tokens_is_real_bpe_not_char_quarter() -> None:
    assert count_tokens("") == 0
    assert count_tokens("abcd") == 1
    assert count_tokens("a" * 40) == 5
    assert estimate_tokens("a" * 40) == 10
    assert count_tokens("hello world") == 2


def test_count_tokens_does_not_crash_on_special_token_lookalike() -> None:
    # tiktoken's default encode() rejects literal special-token strings like "<|endoftext|>"
    # with a ValueError. Retrieved text (docs/chunks/code) can contain this substring
    # incidentally, so count_tokens must not crash on it.
    assert count_tokens("<|endoftext|>") == 7


def test_pack_honors_injected_cost() -> None:
    # seg1 always kept (first). seg2 is 80 'a's: count_tokens=10 (BPE compresses the repeat),
    # estimate_tokens=20 (char/4) -> at budget=15 the two cost functions disagree on whether
    # seg2 fits, so real-token packing includes it while char packing does not.
    segments = [
        Segment(text="b" * 4, source="s1", score=0.9),
        Segment(text="a" * 80, source="s2", score=0.8),
    ]
    real = pack("vector", segments, token_budget=15, cost=count_tokens)
    char = pack("vector", segments, token_budget=15)
    assert [s.source for s in real.segments] == ["s1", "s2"]
    assert [s.source for s in char.segments] == ["s1"]
    assert real.token_estimate == 11
    assert char.token_estimate == 1
    assert real.token_estimate != char.token_estimate


def test_vector_arm_packs_with_real_tokens() -> None:
    from collections.abc import Sequence

    from glyph.baseline.vector import VectorBaseline
    from glyph.embed.port import Vector

    class _FakeEmbedder:
        def embed(self, texts: Sequence[str]) -> list[Vector]:
            return [self._vec(t) for t in texts]

        def _vec(self, text: str) -> Vector:
            low = text.lower()
            return [1.0 if "fogo" in low else 0.0, 1.0 if "goblin" in low else 0.0]

    docs = [
        ("Goblin", "O goblin resiste a fogo."),
        ("Orc", "O orc habita cavernas."),
    ]
    baseline = VectorBaseline(embedder=_FakeEmbedder())
    baseline.index(docs)
    result = baseline.retrieve("fogo", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_graph_arm_packs_with_real_tokens() -> None:
    from collections.abc import Sequence

    from glyph.embed.port import Vector
    from glyph.model.edge import Edge, EdgeType
    from glyph.model.node import Node, NodeType
    from glyph.retrieval.graph import GraphRetriever
    from glyph.store.networkx_store import NetworkXStore

    class _FakeEmbedder:
        def embed(self, texts: Sequence[str]) -> list[Vector]:
            return [self._vec(t) for t in texts]

        def _vec(self, text: str) -> Vector:
            low = text.lower()
            return [1.0 if "fogo" in low else 0.0, 1.0 if "goblin" in low else 0.0]

    nodes = [
        Node(id="goblin", type=NodeType.ENTITY, label="Goblin"),
        Node(
            id="fogo",
            type=NodeType.CONCEPT,
            label="O fogo consome a floresta antiga durante a noite",
        ),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([Edge(src="goblin", dst="fogo", type=EdgeType.RESISTS)])
    retriever = GraphRetriever(store=store, embedder=_FakeEmbedder(), nodes=nodes, hops=1)
    result = retriever.retrieve("fogo", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_hybrid_arm_packs_with_real_tokens() -> None:
    from glyph.model.contract import ContextPack
    from glyph.retrieval.hybrid import HybridRetriever

    class _CannedRetriever:
        def __init__(self, mode: str, segments: list[Segment]) -> None:
            self._mode = mode
            self._segments = segments

        def retrieve(self, query: str, token_budget: int) -> ContextPack:
            return ContextPack(mode=self._mode, segments=self._segments, token_estimate=0)

    graph = _CannedRetriever(
        "graph",
        [Segment(text="O fogo consome a floresta antiga durante a noite", source="a", score=1.0)],
    )
    vector = _CannedRetriever(
        "vector",
        [Segment(text="Um orc habita as cavernas escuras e frias do vale", source="b", score=0.9)],
    )
    result = HybridRetriever(graph, vector).retrieve("q", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_reranked_arm_packs_with_real_tokens() -> None:
    from glyph.model.contract import ContextPack
    from glyph.retrieval.reranked import RerankedRetriever

    class _FakeReranker:
        def rerank(self, query: str, segments: list[Segment], k: int) -> list[Segment]:
            return segments[:k]

    class _FakeRetriever:
        def __init__(self, segments: list[Segment]) -> None:
            self._segments = segments

        def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
            return pack("graph", self._segments, token_budget=token_budget)

    segments = [
        Segment(text="O fogo consome a floresta antiga durante a noite", source="a", score=1.0),
        Segment(text="Um orc habita as cavernas escuras e frias do vale", source="b", score=0.9),
    ]
    retriever = RerankedRetriever(_FakeRetriever(segments), _FakeReranker(), k=2)
    result = retriever.retrieve("q", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_community_arm_packs_with_real_tokens() -> None:
    from collections.abc import Sequence

    from glyph.embed.port import Vector
    from glyph.model.node import Node, NodeType
    from glyph.retrieval.community import CommunityRetriever

    class _KeywordEmbedder:
        _VOCAB = ("fogo", "gelo")

        def embed(self, texts: Sequence[str]) -> list[Vector]:
            return [[1.0 if w in t.lower() else 0.0 for w in self._VOCAB] for t in texts]

    node = Node(
        id="c1",
        type=NodeType.COMMUNITY,
        label="c1",
        attrs={"summary": "O fogo consome a floresta antiga durante a noite"},
    )
    retriever = CommunityRetriever([node], _KeywordEmbedder())
    result = retriever.retrieve("fogo", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_multi_anchor_arm_packs_with_real_tokens() -> None:
    from glyph.embed.port import Vector
    from glyph.model.edge import Edge, EdgeType
    from glyph.model.node import Node, NodeType
    from glyph.retrieval.multi_anchor import MultiAnchorRetriever
    from glyph.store.networkx_store import NetworkXStore

    nodes = [
        Node(
            id="A",
            type=NodeType.ENTITY,
            label="O fogo consome a floresta antiga durante a noite",
        ),
        Node(
            id="B",
            type=NodeType.ENTITY,
            label="Um orc habita as cavernas escuras e frias do vale",
        ),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([Edge(src="A", dst="B", type=EdgeType.RELATES_TO)])

    mapping = {
        "O fogo consome a floresta antiga durante a noite": [1.0, 0.0],
        "Um orc habita as cavernas escuras e frias do vale": [0.0, 1.0],
        "fogo": [1.0, 0.0],
    }

    class _Embedder:
        def embed(self, texts: list[str]) -> list[Vector]:
            return [mapping[t] for t in texts]

    retriever = MultiAnchorRetriever(store, _Embedder(), nodes, hops=1)
    result = retriever.retrieve("fogo", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_path_arm_fewer_than_two_anchors_packs_with_real_tokens() -> None:
    from glyph.embed.port import Vector
    from glyph.model.node import Node, NodeType
    from glyph.retrieval.path import PathRetriever
    from glyph.store.networkx_store import NetworkXStore

    nodes = [
        Node(
            id="A",
            type=NodeType.ENTITY,
            label="O rio corre sereno junto às montanhas altas",
        ),
    ]
    mapping = {
        nodes[0].label: [1.0, 0.0],
        "q": [1.0, 0.0],
    }

    class _Embedder:
        def embed(self, texts: list[str]) -> list[Vector]:
            return [mapping[t] for t in texts]

    retriever = PathRetriever(NetworkXStore(), _Embedder(), nodes)
    result = retriever.retrieve("q", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_path_arm_no_path_between_anchors_packs_with_real_tokens() -> None:
    from glyph.embed.port import Vector
    from glyph.model.node import Node, NodeType
    from glyph.retrieval.path import PathRetriever
    from glyph.store.networkx_store import NetworkXStore

    nodes = [
        Node(
            id="A",
            type=NodeType.ENTITY,
            label="O rio corre sereno junto às montanhas altas",
        ),
        Node(
            id="B",
            type=NodeType.ENTITY,
            label="Um mercado agitado vende especiarias e tecidos raros",
        ),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)  # no edges: A and B are unconnected in the graph

    mapping = {
        nodes[0].label: [1.0, 0.0],
        nodes[1].label: [0.0, 1.0],
        "q": [1.0, 0.0],
    }

    class _Embedder:
        def embed(self, texts: list[str]) -> list[Vector]:
            return [mapping[t] for t in texts]

    retriever = PathRetriever(store, _Embedder(), nodes)
    result = retriever.retrieve("q", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)


def test_path_arm_packs_with_real_tokens() -> None:
    from glyph.embed.port import Vector
    from glyph.model.edge import Edge, EdgeType
    from glyph.model.node import Node, NodeType
    from glyph.retrieval.path import PathRetriever
    from glyph.store.networkx_store import NetworkXStore

    nodes = [
        Node(
            id="A",
            type=NodeType.ENTITY,
            label="O fogo consome a floresta antiga durante a noite",
        ),
        Node(id="B", type=NodeType.ENTITY, label="Um caminho estreito atravessa o vale sombrio"),
        Node(
            id="C",
            type=NodeType.ENTITY,
            label="Um orc habita as cavernas escuras e frias do vale",
        ),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="A", dst="B", type=EdgeType.RELATES_TO),
            Edge(src="B", dst="C", type=EdgeType.RELATES_TO),
        ]
    )
    mapping = {
        nodes[0].label: [1.0, 0.0, 0.0],
        nodes[1].label: [0.0, 1.0, 0.0],
        nodes[2].label: [0.9, 0.0, 0.1],
        "q": [1.0, 0.0, 0.0],
    }

    class _Embedder:
        def embed(self, texts: list[str]) -> list[Vector]:
            return [mapping[t] for t in texts]

    retriever = PathRetriever(store, _Embedder(), nodes)
    result = retriever.retrieve("q", token_budget=1000)
    assert result.token_estimate == sum(count_tokens(s.text) for s in result.segments)
    assert result.token_estimate != sum(estimate_tokens(s.text) for s in result.segments)
