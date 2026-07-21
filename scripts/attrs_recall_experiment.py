"""Issue #21: measure recall@budget delta from including Node.attrs in graph segments.

Deterministic, judge-free (ADR-G8): builds two GraphRetriever arms (baseline vs
include_attrs=True) over the same store/embedder/nodes, retrieves for the
graph_favored subset of the frozen query set, and reports the mean
recall@budget delta plus a keep/revert suggestion.

Env quirk (this environment): if SentenceTransformerEmbedder's HF download
hangs, set HF_HUB_DISABLE_XET=1 before running (hf_xet TLS cert issue).

Usage:
    python3 scripts/attrs_recall_experiment.py
"""

import json
import statistics
from pathlib import Path

from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder
from glyph.eval.recall import recall_at_budget
from glyph.retrieval.graph import GraphRetriever
from glyph.store.networkx_store import NetworkXStore

GRAPH_PATH = "out/monster-manual.json"
QUERIES_PATH = "eval/queries.json"
TOKEN_BUDGET = 1000
KEEP_THRESHOLD = 0.02  # proposed default; a human can override the decision


def main() -> int:
    store = NetworkXStore.load(GRAPH_PATH)
    payload = json.loads(Path(GRAPH_PATH).read_text(encoding="utf-8"))
    nodes = store.subgraph([n["id"] for n in payload["nodes"]], hops=0).nodes
    embedder = SentenceTransformerEmbedder()

    baseline = GraphRetriever(store=store, embedder=embedder, nodes=nodes)
    with_attrs = GraphRetriever(store=store, embedder=embedder, nodes=nodes, include_attrs=True)

    queries = json.loads(Path(QUERIES_PATH).read_text(encoding="utf-8"))["queries"]
    graph_favored = [q for q in queries if q["graph_favored"] is True]

    baseline_scores = []
    attrs_scores = []
    for query in graph_favored:
        relevant = query["relevant_sources"]
        base_pack = baseline.retrieve(query["question"], token_budget=TOKEN_BUDGET)
        attrs_pack = with_attrs.retrieve(query["question"], token_budget=TOKEN_BUDGET)
        baseline_scores.append(recall_at_budget(relevant, {s.source for s in base_pack.segments}))
        attrs_scores.append(recall_at_budget(relevant, {s.source for s in attrs_pack.segments}))

    baseline_mean = statistics.fmean(baseline_scores)
    attrs_mean = statistics.fmean(attrs_scores)
    delta = attrs_mean - baseline_mean
    decision = "KEEP" if delta >= KEEP_THRESHOLD else "REVERT"

    print(f"n={len(graph_favored)} token_budget={TOKEN_BUDGET}")
    print(f"baseline recall@budget:    {baseline_mean:.4f}")
    print(f"attrs-included recall@budget: {attrs_mean:.4f}")
    print(f"delta (attrs - baseline):  {delta:+.4f}")
    print(f"decision ({decision}) via proposed threshold delta >= {KEEP_THRESHOLD} (human can override)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
