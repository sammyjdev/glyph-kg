"""P5.1: GraphContextSource — the entry point AXON's ADR-102 source delegates to.

AXON's GraphContextSource stops re-implementing graph retrieval and calls this instead:
GLYPH is the canonical graph source (ADR-G5), AXON consumes it. This is a thin, stable
facade over graph-aware retrieval — it also folds the "load a persisted graph, list its
nodes, build a retriever" wiring that the scripts repeat into one constructor.
"""

import json
from pathlib import Path

from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack
from glyph.model.node import Node
from glyph.retrieval.graph import GraphRetriever
from glyph.store.networkx_store import NetworkXStore
from glyph.store.port import GraphStore


class GraphContextSource:
    """Turn a query into graph-aware context, over a loaded knowledge graph."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        nodes: list[Node],
        *,
        hops: int = 2,
        anchors: int = 3,
    ) -> None:
        self._retriever = GraphRetriever(
            store=store, embedder=embedder, nodes=nodes, hops=hops, anchors=anchors
        )

    @classmethod
    def from_graph_file(
        cls,
        path: str | Path,
        embedder: Embedder,
        *,
        hops: int = 2,
        anchors: int = 3,
    ) -> "GraphContextSource":
        """Build a source from a persisted NetworkX graph (document or code)."""
        store = NetworkXStore.load(Path(path))
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = [Node.model_validate(n) for n in payload["nodes"]]
        return cls(store, embedder, nodes, hops=hops, anchors=anchors)

    def context(self, query: str, token_budget: int = 1000) -> ContextPack:
        """The graph-aware ContextPack for ``query`` (same contract as every arm)."""
        return self._retriever.retrieve(query, token_budget)
