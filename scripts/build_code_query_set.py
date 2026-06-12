"""P4.4: build the frozen code-domain query set (structural + semantic).

Structural queries (graph-favored) are derived deterministically from the code graph
(out/axon-code.json): callers of moderately-used functions and class inheritance. Semantic
queries (vector-favored) were authored AND verified against the source by a Fable-5
subagent — every oracle relpath was read-checked to exist and be relevant.

The v1 judge is reference-free, so the oracle (relevant_*) is declared metadata, not a
scoring key; its purpose is honesty about what each query targets. The split exists so the
benchmark is fair: structural questions favor the graph arm, semantic ones favor vector.

Usage:
    python3 scripts/build_code_query_set.py            # writes eval/code-queries.json
    python3 scripts/build_code_query_set.py --check    # fail if the committed file is stale
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "out" / "axon-code.json"
SEMANTIC_PATH = ROOT / "eval" / "code-queries-semantic.json"
OUT_PATH = ROOT / "eval" / "code-queries.json"

# Caller-count window: skip 1-2x noise and the SourceRegistry.get (128x) outlier whose
# oracle would be the whole repo; keep functions with a meaningful, tractable caller set.
_MIN_CALLERS, _MAX_CALLERS, _N_CALLER_QUERIES = 5, 25, 6


class Graph:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._label = {n["id"]: n["label"] for n in payload["nodes"]}
        self._type = {n["id"]: n["type"] for n in payload["nodes"]}
        self._edges = payload["edges"]

    def label(self, node_id: str) -> str:
        return self._label.get(node_id, node_id)

    def callers_of(self, target: str) -> list[str]:
        return sorted(e["src"] for e in self._edges if e["type"] == "calls" and e["dst"] == target)

    def call_targets(self) -> list[str]:
        counts = Counter(
            e["dst"]
            for e in self._edges
            if e["type"] == "calls" and self._type.get(e["dst"]) == "function"
        )
        ranked = sorted(
            (nid for nid, c in counts.items() if _MIN_CALLERS <= c <= _MAX_CALLERS),
            key=lambda nid: (-counts[nid], nid),
        )
        return ranked[:_N_CALLER_QUERIES]

    def subclasses_of(self, target: str) -> list[str]:
        return sorted(
            e["src"] for e in self._edges if e["type"] == "inherits" and e["dst"] == target
        )

    def inherited_classes(self) -> list[str]:
        counts = Counter(e["dst"] for e in self._edges if e["type"] == "inherits")
        return sorted(counts, key=lambda nid: (-counts[nid], nid))


def _query(
    qid: str, question: str, sources: list[str], labels: list[str], answer: str, favored: bool
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "category": "structural" if favored else "semantic",
        "graph_favored": favored,
        "relevant_sources": sources,
        "relevant_labels": labels,
        "answer_key": answer,
    }


def build(graph: Graph) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []

    for target in graph.call_targets():
        callers = graph.callers_of(target)
        labels = [graph.label(c) for c in callers]
        queries.append(
            _query(
                f"call-{graph.label(target)}",
                f"Who calls the function `{graph.label(target)}`?",
                callers,
                labels,
                "; ".join(labels),
                favored=True,
            )
        )

    for klass in graph.inherited_classes():
        subs = graph.subclasses_of(klass)
        labels = [graph.label(s) for s in subs]
        queries.append(
            _query(
                f"inherit-{graph.label(klass)}",
                f"Which classes inherit from `{graph.label(klass)}`?",
                subs,
                labels,
                "; ".join(labels),
                favored=True,
            )
        )

    for item in _load_semantic():
        queries.append(
            _query(
                item["id"],
                item["question"],
                item["relevant"],
                item["relevant"],
                item["answer_key"],
                favored=False,
            )
        )

    return {"corpus": "axon-code", "queries": queries}


def _load_semantic() -> list[dict[str, Any]]:
    payload = json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))
    return list(payload["queries"])


def main(argv: list[str]) -> int:
    payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    built = build(Graph(payload))
    rendered = json.dumps(built, ensure_ascii=False, indent=2) + "\n"

    if "--check" in argv:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != rendered:
            print(f"{OUT_PATH} is stale; run scripts/build_code_query_set.py", file=sys.stderr)
            return 1
        print(f"{OUT_PATH} up to date ({len(built['queries'])} queries)")
        return 0

    OUT_PATH.write_text(rendered, encoding="utf-8")
    n_struct = sum(1 for q in built["queries"] if q["graph_favored"])
    print(
        f"wrote {OUT_PATH}: {len(built['queries'])} queries ({n_struct} structural / "
        f"{len(built['queries']) - n_struct} semantic)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
