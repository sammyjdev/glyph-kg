from glyph.model.edge import Edge, EdgeType
from glyph.model.graph import Path, Subgraph
from glyph.model.node import Node, NodeType


def _node(nid: str) -> Node:
    return Node(id=nid, type=NodeType.ENTITY, label=nid)


def test_subgraph_holds_nodes_and_edges() -> None:
    nodes = [_node("a"), _node("b")]
    edges = [Edge(src="a", dst="b", type=EdgeType.RELATES_TO)]
    sg = Subgraph(nodes=nodes, edges=edges)
    assert list(sg.nodes) == nodes
    assert list(sg.edges) == edges


def test_subgraph_defaults_to_empty() -> None:
    sg = Subgraph()
    assert list(sg.nodes) == []
    assert list(sg.edges) == []


def test_subgraph_round_trips_through_json() -> None:
    sg = Subgraph(nodes=[_node("a")], edges=[Edge(src="a", dst="a", type=EdgeType.MENTIONS)])
    assert Subgraph.model_validate_json(sg.model_dump_json()) == sg


def test_path_is_ordered_sequence_of_node_ids() -> None:
    path = Path(nodes=["a", "b", "c"])
    assert list(path.nodes) == ["a", "b", "c"]


def test_path_round_trips_through_json() -> None:
    path = Path(nodes=["a", "b"])
    assert Path.model_validate_json(path.model_dump_json()) == path
