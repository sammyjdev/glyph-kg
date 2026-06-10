import pytest
from pydantic import ValidationError

from glyph.model.edge import Edge, EdgeType


def test_edge_connects_two_node_ids() -> None:
    edge = Edge(src="goblin", dst="fire", type=EdgeType.RESISTS)
    assert edge.src == "goblin"
    assert edge.dst == "fire"
    assert edge.type is EdgeType.RESISTS
    assert edge.attrs == {}


def test_edge_carries_arbitrary_attrs() -> None:
    edge = Edge(src="a", dst="b", type=EdgeType.CALLS, attrs={"line": 42})
    assert edge.attrs == {"line": 42}


def test_edgetype_separates_code_and_document_domains() -> None:
    code = {
        EdgeType.DEFINES,
        EdgeType.IMPORTS,
        EdgeType.CALLS,
        EdgeType.INHERITS,
        EdgeType.REFERENCES,
    }
    document = {EdgeType.RELATES_TO, EdgeType.MENTIONS, EdgeType.REQUIRES, EdgeType.RESISTS}
    assert code.isdisjoint(document)
    assert code | document <= set(EdgeType)


def test_edge_is_frozen() -> None:
    edge = Edge(src="a", dst="b", type=EdgeType.MENTIONS)
    with pytest.raises(ValidationError):
        edge.dst = "c"  # type: ignore[misc]


def test_edge_round_trips_through_json() -> None:
    edge = Edge(src="a", dst="b", type=EdgeType.REQUIRES, attrs={"weight": 0.8})
    assert Edge.model_validate_json(edge.model_dump_json()) == edge


def test_edge_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Edge(src="a", dst="b", type="teleports")  # type: ignore[arg-type]
