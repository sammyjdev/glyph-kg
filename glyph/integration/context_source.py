"""GraphContextSource — the stable product boundary for graph-aware retrieval (dec-g6).

This is the named entry point external consumers (AXON, future clients) depend on instead
of wiring ``GraphRetriever`` themselves. It is a thin facade that satisfies the
:class:`~glyph.retrieval.port.Retriever` port (``retrieve(query, token_budget) -> ContextPack``),
so it drops in anywhere a ``Retriever`` is expected, and the boundary can evolve (hops,
anchors, hybrid fusion) without breaking callers.

Two entry points, same object:

- ``GraphContextSource(store, embedder, nodes)`` — **in-memory**: the caller already holds a
  ``GraphStore`` and its node list (e.g. AXON builds them from its SQLite graph).
- ``GraphContextSource.from_graph_file(path, embedder)`` — **persisted**: load a NetworkX
  graph (document or code) from disk, folding the load + node-listing + wiring into one call.
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

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        """The graph-aware ContextPack for ``query`` — the ``Retriever`` port contract."""
        return self._retriever.retrieve(query, token_budget)
