"""Tests for the generic personal-notes extraction schema and merge function."""

from __future__ import annotations

from glyph.extract.document.schema_notes import (
    NotesEntity,
    NotesExtractionResult,
    NotesRelation,
    merge_notes,
)
from glyph.model.edge import EdgeType
from glyph.model.node import NodeType


def _result(
    entities: list[NotesEntity] | None = None,
    relations: list[NotesRelation] | None = None,
) -> NotesExtractionResult:
    return NotesExtractionResult(
        entities=entities or [],
        relations=relations or [],
    )


# ---------------------------------------------------------------------------
# Entity mapping
# ---------------------------------------------------------------------------


def test_merge_notes_person_maps_to_entity_node() -> None:
    nodes, _ = merge_notes([_result([NotesEntity(name="Alice", kind="person")])])
    by_id = {n.id: n for n in nodes}
    assert by_id["alice"].type is NodeType.ENTITY


def test_merge_notes_project_maps_to_entity_node() -> None:
    nodes, _ = merge_notes([_result([NotesEntity(name="Glyph", kind="project")])])
    by_id = {n.id: n for n in nodes}
    assert by_id["glyph"].type is NodeType.ENTITY


def test_merge_notes_concept_maps_to_concept_node() -> None:
    nodes, _ = merge_notes([_result([NotesEntity(name="GraphRAG", kind="concept")])])
    by_id = {n.id: n for n in nodes}
    assert by_id["graphrag"].type is NodeType.CONCEPT


def test_merge_notes_note_maps_to_section_node() -> None:
    nodes, _ = merge_notes([_result([NotesEntity(name="Daily Note", kind="note")])])
    by_id = {n.id: n for n in nodes}
    assert by_id["daily note"].type is NodeType.SECTION


def test_merge_notes_source_maps_to_concept_node() -> None:
    nodes, _ = merge_notes([_result([NotesEntity(name="Paper XYZ", kind="source")])])
    by_id = {n.id: n for n in nodes}
    assert by_id["paper xyz"].type is NodeType.CONCEPT


def test_merge_notes_optional_attrs_included_when_present() -> None:
    entity = NotesEntity(name="Alice", kind="person", description="My colleague", url="https://x")
    nodes, _ = merge_notes([_result([entity])])
    by_id = {n.id: n for n in nodes}
    assert by_id["alice"].attrs["description"] == "My colleague"
    assert by_id["alice"].attrs["url"] == "https://x"


def test_merge_notes_optional_attrs_omitted_when_none() -> None:
    entity = NotesEntity(name="Bob", kind="person")
    nodes, _ = merge_notes([_result([entity])])
    assert nodes[0].attrs == {}


# ---------------------------------------------------------------------------
# Relation mapping
# ---------------------------------------------------------------------------


def test_merge_notes_relates_to_predicate() -> None:
    r = NotesRelation(subject="Alice", predicate="RELATES_TO", object="Bob")
    _, edges = merge_notes(
        [
            _result(
                entities=[
                    NotesEntity(name="Alice", kind="person"),
                    NotesEntity(name="Bob", kind="person"),
                ],
                relations=[r],
            )
        ]
    )
    assert any(e.type is EdgeType.RELATES_TO for e in edges)


def test_merge_notes_mentions_predicate() -> None:
    r = NotesRelation(subject="Note A", predicate="MENTIONS", object="Alice")
    _, edges = merge_notes(
        [
            _result(
                entities=[NotesEntity(name="Note A", kind="note")],
                relations=[r],
            )
        ]
    )
    assert any(e.type is EdgeType.MENTIONS for e in edges)


def test_merge_notes_unlisted_subject_gets_concept_node() -> None:
    """A subject not in entities is auto-created as a CONCEPT node."""
    r = NotesRelation(subject="Unknown", predicate="RELATES_TO", object="Alice")
    nodes, _ = merge_notes(
        [_result(entities=[NotesEntity(name="Alice", kind="person")], relations=[r])]
    )
    by_id = {n.id: n for n in nodes}
    assert "unknown" in by_id
    assert by_id["unknown"].type is NodeType.CONCEPT


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_merge_notes_deduplicates_nodes_across_results() -> None:
    r = _result([NotesEntity(name="Alice", kind="person")])
    nodes, _ = merge_notes([r, r])
    assert len([n for n in nodes if n.id == "alice"]) == 1


def test_merge_notes_deduplicates_edges_across_results() -> None:
    rel = NotesRelation(subject="Alice", predicate="RELATES_TO", object="Bob")
    r = _result(
        entities=[
            NotesEntity(name="Alice", kind="person"),
            NotesEntity(name="Bob", kind="person"),
        ],
        relations=[rel],
    )
    _, edges = merge_notes([r, r])
    assert len(edges) == 1


def test_merge_notes_normalises_ids() -> None:
    entity = NotesEntity(name="  Alice Smith ", kind="person")
    nodes, _ = merge_notes([_result([entity])])
    assert nodes[0].id == "alice smith"
    assert nodes[0].label == "Alice Smith"


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_notes_extraction_result_parses_from_dict() -> None:
    data = {
        "entities": [{"name": "Glyph", "kind": "project"}],
        "relations": [{"subject": "Glyph", "predicate": "DEPENDS_ON", "object": "NetworkX"}],
    }
    result = NotesExtractionResult.model_validate(data)
    assert result.entities[0].name == "Glyph"
    assert result.relations[0].predicate == "DEPENDS_ON"
