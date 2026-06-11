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
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import networkx as nx

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
        digest = hashlib.sha1("\n".join(self.members).encode("utf-8")).hexdigest()[:12]
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
