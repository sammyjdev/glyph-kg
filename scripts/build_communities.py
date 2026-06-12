"""P7 build: augment a persisted graph with detected + summarized communities.

Loads a graph, runs Louvain community detection (seeded, deterministic), summarizes each
community via an injected OpenAI-compatible LLM (NVIDIA NIM by default — free), and writes
a new graph with COMMUNITY nodes (attrs: title/summary) + CONTAINS edges. The result feeds
the global benchmark arm (CommunityRetriever).

Usage:
    NVIDIA_NIM_API_KEY=... python3 scripts/build_communities.py out/axon-code.json \\
        out/axon-code-communities.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from glyph.eval.judge import _urllib_post
from glyph.model.node import Node
from glyph.retrieval.community import (
    CommunitySummary,
    detect_communities,
    summarize_communities,
    to_graph_elements,
)
from glyph.store.networkx_store import NetworkXStore

_NIM_BASE = "https://integrate.api.nvidia.com/v1"
_NIM_MODEL = "meta/llama-3.3-70b-instruct"


class _OpenAICompatSummarizer:
    """A CommunitySummarizer over any OpenAI-compatible endpoint (NIM default, free).

    GLYPH owns the member prompt (`summarize_communities`); this adapter only adds the
    JSON-shape instruction, calls the model, and parses {title, summary}.
    """

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key

    def summarize(self, prompt: str) -> CommunitySummary:
        content = (
            prompt + '\n\nReturn ONLY JSON: {"title": "<short thematic title>", '
            '"summary": "<1-2 sentence summary>"}'
        )
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 400,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = _urllib_post(self._url, payload, headers, 120.0)
        parsed = json.loads(body["choices"][0]["message"]["content"])
        return CommunitySummary(title=str(parsed["title"]), summary=str(parsed["summary"]))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    import os

    graph_path, out_path = argv
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("NVIDIA_NIM_API_KEY not set", file=sys.stderr)
        return 2

    store = NetworkXStore.load(graph_path)
    payload = json.loads(open(graph_path, encoding="utf-8").read())
    nodes = [Node.model_validate(n) for n in payload["nodes"]]

    communities = detect_communities(store, nodes, seed=0)
    print(f"detected {len(communities)} communities", flush=True)

    # qualified ids carry file+symbol context — richer summaries than bare labels
    member_text = {n.id: n.id for n in nodes}
    summarizer = _OpenAICompatSummarizer(_NIM_MODEL, api_key, _NIM_BASE)
    comm_nodes = summarize_communities(communities, member_text, summarizer)
    _, contains_edges = to_graph_elements(communities)

    store.upsert_nodes(comm_nodes)
    store.upsert_edges(contains_edges)
    store.save(out_path)

    sizes = Counter(len(c.members) for c in communities)
    print(f"wrote {out_path}: +{len(comm_nodes)} COMMUNITY nodes, +{len(contains_edges)} CONTAINS")
    print(f"  community sizes (members->count): {dict(sorted(sizes.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
