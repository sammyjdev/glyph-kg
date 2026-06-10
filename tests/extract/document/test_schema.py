from glyph.extract.document.schema import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    merge,
)
from glyph.model.edge import EdgeType
from glyph.model.node import NodeType


def _result() -> ExtractionResult:
    return ExtractionResult(
        entities=[
            ExtractedEntity(name="Goblin", kind="creature", creature_type="humanoide"),
            ExtractedEntity(name="fogo", kind="concept"),
        ],
        relations=[ExtractedRelation(subject="Goblin", predicate="RESISTS", object="fogo")],
    )


def test_merge_maps_entities_to_typed_nodes() -> None:
    nodes, _ = merge([_result()])
    by_id = {n.id: n for n in nodes}
    assert by_id["goblin"].type is NodeType.ENTITY
    assert by_id["goblin"].label == "Goblin"
    assert by_id["goblin"].attrs == {"creature_type": "humanoide"}
    assert by_id["fogo"].type is NodeType.CONCEPT


def test_merge_maps_relations_to_typed_edges() -> None:
    _, edges = merge([_result()])
    (edge,) = edges
    assert edge.src == "goblin"
    assert edge.dst == "fogo"
    assert edge.type is EdgeType.RESISTS


def test_merge_creates_concept_node_for_unlisted_relation_object() -> None:
    result = ExtractionResult(
        entities=[ExtractedEntity(name="Orc", kind="creature")],
        relations=[ExtractedRelation(subject="Orc", predicate="INHABITS", object="cavernas")],
    )
    nodes, _ = merge([result])
    by_id = {n.id: n for n in nodes}
    assert by_id["cavernas"].type is NodeType.CONCEPT


def test_merge_dedupes_nodes_and_edges_across_results() -> None:
    nodes, edges = merge([_result(), _result()])
    assert len([n for n in nodes if n.id == "goblin"]) == 1
    assert len(edges) == 1


def test_merge_normalizes_id_by_lowercasing_and_collapsing_space() -> None:
    result = ExtractionResult(
        entities=[ExtractedEntity(name="  Verme   Púrpura ", kind="creature")],
        relations=[],
    )
    (node,) = merge([result])[0]
    assert node.id == "verme púrpura"
    assert node.label == "Verme   Púrpura"
