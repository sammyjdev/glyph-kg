"""P1.4 cost gate: extract a book, report cost/latency/volume, persist the graph.

Usage:
    ANTHROPIC_API_KEY=... python3 scripts/extract_book.py "<book.pdf>" out/graph.json

Runs the live Anthropic API and costs money. Do not run without explicit approval.
"""

import sys
import time
from pathlib import Path

from glyph.extract.document.cost import summarize
from glyph.extract.document.extractor import DocumentExtractor
from glyph.model.node import NodeType
from glyph.store.networkx_store import NetworkXStore


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    book_path, out_path = argv[1], argv[2]

    started = time.monotonic()
    nodes, edges, usages = DocumentExtractor().extract_with_usage(book_path)
    elapsed = time.monotonic() - started

    report = summarize(usages)
    creatures = [n for n in nodes if n.type is NodeType.ENTITY]

    print(f"book:     {book_path}")
    print(f"chunks:   {report.chunks}")
    print(f"nodes:    {len(nodes)} ({len(creatures)} creatures)")
    print(f"edges:    {len(edges)}")
    print(f"tokens:   in={report.input_tokens} out={report.output_tokens}")
    print(f"cost:     ${report.cost_usd:.4f}")
    print(f"latency:  {elapsed:.1f}s ({elapsed / max(report.chunks, 1):.2f}s/chunk)")

    print("\nsample creatures (up to 10):")
    for node in creatures[:10]:
        out = [e for e in edges if e.src == node.id]
        rels = ", ".join(f"{e.type.value} {e.dst}" for e in out[:5])
        print(f"  - {node.label}: {rels or '(no relations)'}")

    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(edges)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    store.save(Path(out_path))
    print(f"\npersisted graph -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
