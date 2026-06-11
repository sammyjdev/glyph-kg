"""GraphContextSource is the stable seam AXON delegates to: query -> graph context."""

from collections.abc import Sequence
from pathlib import Path

from glyph.embed.port import Vector
from glyph.integration import GraphContextSource
from glyph.model.contract import ContextPack
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


class FakeEmbedder:
    """Deterministic word-overlap vectors over a tiny fixed vocabulary."""

    _VOCAB = ("goblin", "dragon", "fire", "cold", "resists")

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            lowered = text.lower()
            out.append([1.0 if word in lowered else 0.0 for word in self._VOCAB])
        return out


def _nodes() -> list[Node]:
    return [
        Node(id="goblin", type=NodeType.ENTITY, label="Goblin"),
        Node(id="dragon", type=NodeType.ENTITY, label="Dragon"),
        Node(id="fire", type=NodeType.CONCEPT, label="fire"),
    ]


def _edges() -> list[Edge]:
    return [Edge(src="goblin", dst="fire", type=EdgeType.RESISTS)]


def _store() -> NetworkXStore:
    store = NetworkXStore()
    store.upsert_nodes(_nodes())
    store.upsert_edges(_edges())
    return store


def test_satisfies_the_retriever_port() -> None:
    # The product boundary IS a Retriever: drops in anywhere the port is expected.
    source = GraphContextSource(_store(), FakeEmbedder(), _nodes())
    assert isinstance(source, Retriever)


def test_retrieve_returns_a_graph_context_pack() -> None:
    source = GraphContextSource(_store(), FakeEmbedder(), _nodes(), anchors=2)
    pack = source.retrieve("which creature resists fire?")
    assert isinstance(pack, ContextPack)
    assert pack.mode == "graph"
    assert pack.segments  # at least one segment


def test_retrieve_honors_token_budget() -> None:
    source = GraphContextSource(_store(), FakeEmbedder(), _nodes())
    pack = source.retrieve("goblin", token_budget=5)
    assert pack.token_estimate <= max(5, len(pack.segments[0].text))  # always keeps the first


def test_from_graph_file_loads_and_serves(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    _store().save(path)
    source = GraphContextSource.from_graph_file(path, FakeEmbedder(), anchors=3)
    pack = source.retrieve("goblin resists fire")
    assert isinstance(pack, ContextPack)
    assert any("goblin" in s.source for s in pack.segments)
