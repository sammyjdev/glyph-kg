"""P4.3: extract a code knowledge graph from a source tree and persist it.

Deterministic (tree-sitter): the same source at a fixed SHA yields the same graph.
Self-hosting example — extract GLYPH's own package:

    python3 scripts/extract_code.py glyph out/glyph-code.json

Requires: pip install -e ".[code]"
"""

import json
import sys
from collections import Counter
from pathlib import Path

from glyph.extract.code import CodeExtractor


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    source, out_path = argv[1], argv[2]

    nodes, edges = CodeExtractor().extract(source)
    payload = {
        "nodes": [n.model_dump(mode="json") for n in nodes],
        "edges": [e.model_dump(mode="json") for e in edges],
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")

    node_kinds = Counter(n.type.value for n in nodes)
    edge_kinds = Counter(e.type.value for e in edges)
    print(f"wrote {out_path}: {len(nodes)} nodes, {len(edges)} edges")
    print(f"  nodes: {dict(node_kinds)}")
    print(f"  edges: {dict(edge_kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
