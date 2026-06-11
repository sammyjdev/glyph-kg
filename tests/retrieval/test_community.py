"""P7 community detection + graph elements (deterministic, structural-only)."""

from collections.abc import Iterable, Sequence

from glyph.embed.port import Vector
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.community import (
    Community,
    CommunityRetriever,
    CommunitySummary,
    detect_communities,
    summarize_communities,
    to_graph_elements,
)
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


def _fn(nid: str) -> Node:
    return Node(id=nid, type=NodeType.FUNCTION, label=nid)


def _members(comms: Iterable[Community]) -> list[list[str]]:
    return sorted(sorted(c.members) for c in comms)


def _two_triangles() -> tuple[NetworkXStore, list[Node]]:
    # two dense triangles {a,b,c} and {d,e,f} joined by a single bridge c-d
    nodes = [_fn(x) for x in "abcdef"]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    pairs = [("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f"), ("c", "d")]
    store.upsert_edges([Edge(src=u, dst=v, type=EdgeType.CALLS) for u, v in pairs])
    return store, nodes


def test_detect_partitions_two_clusters() -> None:
    store, nodes = _two_triangles()
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a", "b", "c"], ["d", "e", "f"]]


def test_detect_ignores_preexisting_community_nodes() -> None:
    store, nodes = _two_triangles()
    # a stale overlay from a previous build must not be treated as a member
    store.upsert_nodes([Node(id="comm", type=NodeType.COMMUNITY, label="old")])
    store.upsert_edges([Edge(src="comm", dst="a", type=EdgeType.CONTAINS)])
    nodes = [*nodes, Node(id="comm", type=NodeType.COMMUNITY, label="old")]
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a", "b", "c"], ["d", "e", "f"]]


def test_detect_clusters_on_structural_edges_only() -> None:
    # nodes joined only by a document edge (MENTIONS) have no structural links,
    # so each is its own community — ADRs/entities never merge with code.
    nodes = [_fn("a"), _fn("b"), _fn("c")]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="a", dst="b", type=EdgeType.MENTIONS),
            Edge(src="b", dst="c", type=EdgeType.MENTIONS),
        ]
    )
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a"], ["b"], ["c"]]


def test_detect_is_reproducible_with_seed() -> None:
    store, nodes = _two_triangles()
    run1 = [c.id for c in detect_communities(store, nodes, seed=0)]
    run2 = [c.id for c in detect_communities(store, nodes, seed=0)]
    assert run1 == run2
    assert all(cid.startswith("community:") for cid in run1)


def test_detect_ignores_self_loops() -> None:
    # a recursive call (CALLS itself) is a self-loop in the graph; it must be
    # skipped, not break detection.
    nodes = [_fn("a"), _fn("b"), _fn("c")]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(
        [
            Edge(src="a", dst="a", type=EdgeType.CALLS),  # recursion
            Edge(src="a", dst="b", type=EdgeType.CALLS),
            Edge(src="b", dst="c", type=EdgeType.CALLS),
            Edge(src="a", dst="c", type=EdgeType.CALLS),
        ]
    )
    comms = detect_communities(store, nodes, seed=0)
    assert _members(comms) == [["a", "b", "c"]]


class _StubSummarizer:
    """Fake summarizer (zero LLM cost): echoes the prompt so tests can inspect it."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def summarize(self, prompt: str) -> CommunitySummary:
        self.prompts.append(prompt)
        return CommunitySummary(title="Fire faction", summary="Creatures around fire.")


def test_summarize_fills_summary_title_and_label() -> None:
    comms = [Community(members=("dragon", "salamander"))]
    member_text = {"dragon": "Red Dragon", "salamander": "Salamander"}
    (node,) = summarize_communities(comms, member_text, _StubSummarizer())
    assert node.type is NodeType.COMMUNITY
    assert node.id == comms[0].id
    assert node.attrs["summary"] == "Creatures around fire."
    assert node.attrs["title"] == "Fire faction"
    assert node.attrs["members"] == 2
    assert node.label == "Fire faction"  # the thematic one-liner is the label


def test_summarize_prompt_is_built_from_member_text() -> None:
    summarizer = _StubSummarizer()
    summarize_communities([Community(members=("dragon",))], {"dragon": "Red Dragon"}, summarizer)
    assert "Red Dragon" in summarizer.prompts[0]  # GLYPH builds the prompt from members


def test_to_graph_elements_builds_community_nodes_and_contains_edges() -> None:
    comms = [Community(members=("a", "b", "c"))]
    comm_nodes, edges = to_graph_elements(comms)
    (node,) = comm_nodes
    assert node.type is NodeType.COMMUNITY
    assert node.id == comms[0].id
    assert node.attrs["members"] == 3
    assert {(e.src, e.dst, e.type) for e in edges} == {
        (node.id, "a", EdgeType.CONTAINS),
        (node.id, "b", EdgeType.CONTAINS),
        (node.id, "c", EdgeType.CONTAINS),
    }


class _KeywordEmbedder:
    """Deterministic keyword embedder over a tiny vocabulary [fire, ice, goblin]."""

    _VOCAB = ("fire", "ice", "goblin")

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [[1.0 if w in t.lower() else 0.0 for w in self._VOCAB] for t in texts]


def _comm_node(cid: str, summary: str) -> Node:
    return Node(id=cid, type=NodeType.COMMUNITY, label=cid, attrs={"summary": summary})


def test_community_retriever_satisfies_the_port() -> None:
    retriever = CommunityRetriever([_comm_node("c1", "about fire")], _KeywordEmbedder())
    assert isinstance(retriever, Retriever)


def test_retrieve_ranks_the_relevant_community_first() -> None:
    nodes = [_comm_node("fire", "creatures of fire"), _comm_node("ice", "creatures of ice")]
    retriever = CommunityRetriever(nodes, _KeywordEmbedder())
    result = retriever.retrieve("fire monsters", token_budget=1000)
    assert result.mode == "community"
    assert result.segments[0].source == "fire"
    assert result.segments[0].text == "creatures of fire"


def test_retrieve_is_empty_when_there_are_no_communities() -> None:
    result = CommunityRetriever([], _KeywordEmbedder()).retrieve("anything")
    assert result.mode == "community"
    assert result.segments == []
