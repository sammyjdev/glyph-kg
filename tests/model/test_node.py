import pytest
from pydantic import ValidationError

from glyph.model.node import Node, NodeType


def test_node_requires_id_type_label() -> None:
    node = Node(id="m.func", type=NodeType.FUNCTION, label="func")
    assert node.id == "m.func"
    assert node.type is NodeType.FUNCTION
    assert node.label == "func"
    assert node.attrs == {}


def test_node_carries_arbitrary_attrs() -> None:
    node = Node(id="goblin", type=NodeType.ENTITY, label="Goblin", attrs={"book": "MM", "page": 12})
    assert node.attrs == {"book": "MM", "page": 12}


def test_nodetype_covers_code_and_document_domains() -> None:
    code = {NodeType.FILE, NodeType.MODULE, NodeType.CLASS, NodeType.FUNCTION}
    document = {NodeType.ENTITY, NodeType.CONCEPT, NodeType.SECTION}
    assert code.isdisjoint(document)
    assert code | document <= set(NodeType)


def test_node_is_frozen() -> None:
    node = Node(id="x", type=NodeType.CONCEPT, label="X")
    with pytest.raises(ValidationError):
        node.label = "Y"  # type: ignore[misc]


def test_node_round_trips_through_json() -> None:
    node = Node(id="x", type=NodeType.SECTION, label="Combat", attrs={"level": 3})
    assert Node.model_validate_json(node.model_dump_json()) == node


def test_node_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Node(id="x", type="dragon", label="X")  # type: ignore[arg-type]
