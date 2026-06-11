"""Run the three retrieval arms over the Monster Manual graph for manual inspection.

Usage:
    python3 scripts/retrieve_demo.py <graph.json> <book.pdf> "<query>"

Uses local sentence-transformers embeddings (no API cost). Requires:
    pip install -e ".[document,retrieval,embeddings]"
"""

import sys

from glyph.baseline.vector import VectorBaseline
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder
from glyph.extract.document import chunk, pdf
from glyph.model.contract import ContextPack
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.hybrid import HybridRetriever
from glyph.store.networkx_store import NetworkXStore


def _show(name: str, result: ContextPack) -> None:
    print(f"\n=== {name} ({result.token_estimate} tok, {len(result.segments)} segments) ===")
    for segment in result.segments[:8]:
        print(f"  [{segment.score:.3f}] {segment.source}: {segment.text[:110]}")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    graph_path, book_path, query = argv[1], argv[2], argv[3]

    store = NetworkXStore.load(graph_path)
    nodes = store.subgraph(_all_ids(graph_path), hops=0).nodes

    documents = [
        (piece.label, piece.text)
        for piece in chunk.by_creature(pdf.load(book_path))
        if chunk.is_creature(piece)
    ]

    embedder = SentenceTransformerEmbedder()
    graph = GraphRetriever(store=store, embedder=embedder, nodes=nodes)
    vector = VectorBaseline(embedder=embedder)
    vector.index(documents)
    hybrid = HybridRetriever(graph, vector)

    print(f"query: {query!r}")
    _show("graph", graph.retrieve(query))
    _show("vector", vector.retrieve(query))
    _show("hybrid", hybrid.retrieve(query))
    return 0


def _all_ids(graph_path: str) -> list[str]:
    import json
    from pathlib import Path

    payload = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    return [node["id"] for node in payload["nodes"]]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
