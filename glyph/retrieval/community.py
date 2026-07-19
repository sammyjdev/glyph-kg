"""P7: community detection over the structural graph (dec-g7).

Detects communities (Louvain, networkx-native, zero new dependency) over the
*structural* projection of the graph, deterministically. The global retrieval axis
summarizes these communities and serves the summaries for thematic queries; this
module owns detection + the graph elements, summarization, and the retriever.

Determinism: Louvain is stochastic, so a fixed ``seed`` is required — and networkx's
result also depends on iteration order, so we feed nodes/edges in sorted order. Same
graph + seed → same communities → reproducible artifact (GLYPH's invariant).
"""

import hashlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import networkx as nx

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, pack
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.store.port import GraphStore

# Cluster on code structure only. Document/decision edges (MENTIONS, RELATES_TO, ...)
# would mix unrelated entities into a community, so they are excluded by default.
STRUCTURAL_EDGES: frozenset[EdgeType] = frozenset(
    {
        EdgeType.DEFINES,
        EdgeType.IMPORTS,
        EdgeType.CALLS,
        EdgeType.INHERITS,
        EdgeType.REFERENCES,
    }
)


@dataclass(frozen=True)
class Community:
    """A detected community: its sorted member node ids and a stable derived id."""

    members: tuple[str, ...]

    @property
    def id(self) -> str:
        # Stable across runs (hashlib, not the salted builtin hash): unchanged
        # communities keep their id between rebuilds, so summarization can skip them.
        digest = hashlib.sha1(
            "\n".join(self.members).encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:12]
        return f"community:{digest}"


def detect_communities(
    store: GraphStore,
    nodes: Sequence[Node],
    *,
    seed: int,
    edge_types: Collection[EdgeType] = STRUCTURAL_EDGES,
) -> list[Community]:
    """Partition the structural subgraph into communities (Louvain, seeded)."""
    member_ids = sorted(n.id for n in nodes if n.type is not NodeType.COMMUNITY)
    id_set = set(member_ids)
    exclude_edges = frozenset(set(EdgeType) - set(edge_types))
    structural = store.subgraph(
        member_ids,
        hops=1,
        exclude_node_types={NodeType.COMMUNITY},
        exclude_edge_types=exclude_edges,
    )

    graph = nx.Graph()
    graph.add_nodes_from(member_ids)  # sorted → deterministic node order
    for edge in sorted(structural.edges, key=lambda e: (e.src, e.dst, e.type.value)):
        if edge.src in id_set and edge.dst in id_set and edge.src != edge.dst:
            graph.add_edge(edge.src, edge.dst)

    raw = nx.community.louvain_communities(graph, seed=seed)
    communities = [Community(members=tuple(sorted(group))) for group in raw]
    communities.sort(key=lambda c: c.members)
    return communities


@dataclass(frozen=True)
class CommunitySummary:
    """A community's thematic one-line title and short prose summary."""

    title: str
    summary: str


class CommunitySummarizer(Protocol):
    """Injected LLM boundary: GLYPH builds the prompt, the caller runs the model.

    Mirrors ``DocumentExtractor``'s ``LLMExtractor`` — GLYPH owns *what* to summarize
    and the prompt; the provider/API key stay outside the library (AXON injects them).
    """

    def summarize(self, prompt: str) -> CommunitySummary: ...


_PROMPT_HEADER = (
    "Summarize this group of related graph nodes as one coherent theme. "
    "Give a short thematic title and a 1-2 sentence summary.\n\nMembers:\n"
)


def _summary_prompt(labels: Sequence[str]) -> str:
    return _PROMPT_HEADER + "\n".join(f"- {label}" for label in labels)


def summarize_communities(
    communities: Sequence[Community],
    member_text: Mapping[str, str],
    summarizer: CommunitySummarizer,
) -> list[Node]:
    """Summarize each community via the injected LLM into a COMMUNITY node.

    ``member_text`` maps a member id to the text shown to the model (labels for code).
    To save cost on rebuilds, pass only new/changed communities — ids are stable
    (``Community.id``), so the caller can skip the ones already summarized.
    """
    out: list[Node] = []
    for community in communities:
        labels = [member_text.get(member, member) for member in community.members]
        result = summarizer.summarize(_summary_prompt(labels))
        out.append(
            Node(
                id=community.id,
                type=NodeType.COMMUNITY,
                label=result.title,
                attrs={
                    "members": len(community.members),
                    "title": result.title,
                    "summary": result.summary,
                },
            )
        )
    return out


def to_graph_elements(communities: Sequence[Community]) -> tuple[list[Node], list[Edge]]:
    """COMMUNITY nodes + CONTAINS edges (community → each member), for upsert.

    Labels default to the id; ``summarize_communities`` fills attrs ``summary``/``title``.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []
    for community in communities:
        nodes.append(
            Node(
                id=community.id,
                type=NodeType.COMMUNITY,
                label=community.id,
                attrs={"members": len(community.members)},
            )
        )
        edges.extend(
            Edge(src=community.id, dst=member, type=EdgeType.CONTAINS)
            for member in community.members
        )
    return nodes, edges


class CommunityRetriever:
    """Global-axis retriever: rank community summaries against a thematic query.

    Satisfies the ``Retriever`` port (``retrieve(query, token_budget) -> ContextPack``)
    like every arm, but answers "how is this organized?" from community summaries
    instead of expanding a local neighborhood — so it never traverses the graph.
    """

    def __init__(self, community_nodes: Sequence[Node], embedder: Embedder, top_k: int = 5) -> None:
        self._embedder = embedder
        self._top_k = top_k
        self._summary = {node.id: str(node.attrs.get("summary", "")) for node in community_nodes}
        self._index = InMemoryVectorIndex()
        ids = list(self._summary)
        if ids:
            vectors = embedder.embed([self._summary[cid] for cid in ids])
            for cid, vector in zip(ids, vectors, strict=True):
                self._index.add(cid, vector)

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        if not self._summary:
            return pack("community", [], token_budget)
        query_vector = self._embedder.embed([query])[0]
        hits = self._index.search(query_vector, self._top_k)
        segments = [
            Segment(text=self._summary[cid], source=cid, score=score) for cid, score in hits
        ]
        segments.sort(key=lambda s: (-s.score, s.source))  # stable, score-desc
        return pack("community", segments, token_budget)
