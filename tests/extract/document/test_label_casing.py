"""Label-casing normalization for merge()/merge_notes() (issue #18).

Ids are already normalized (lowercased, whitespace-collapsed); this covers the
label itself, which used to keep whichever surface form was seen first.

Test plan:
1. Most-frequent form wins, for both merge() and merge_notes().
2. Deterministic tie-break: 1-vs-1 count keeps the first-seen form, repeatably.
3. Id/edge invariance: casing-only differences never change the node id set or
   edge set, and the chosen label is always one of the surface forms actually
   seen (never synthesized).
"""

from glyph.extract.document.schema import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    merge,
)
from glyph.extract.document.schema_notes import (
    NotesEntity,
    NotesExtractionResult,
    merge_notes,
)


def _entity_result(name: str) -> ExtractionResult:
    return ExtractionResult(entities=[ExtractedEntity(name=name, kind="creature")], relations=[])


def _notes_result(name: str) -> NotesExtractionResult:
    return NotesExtractionResult(entities=[NotesEntity(name=name, kind="person")], relations=[])


# ---------------------------------------------------------------------------
# 1. Most-frequent surface form wins
# ---------------------------------------------------------------------------


def test_merge_picks_most_frequent_label_across_chunks() -> None:
    results = [_entity_result("ANKHEG"), _entity_result("Ankheg"), _entity_result("Ankheg")]
    (node,) = merge(results)[0]
    assert node.label == "Ankheg"


def test_merge_notes_picks_most_frequent_label_across_chunks() -> None:
    results = [_notes_result("deva"), _notes_result("Deva"), _notes_result("Deva")]
    (node,) = merge_notes(results)[0]
    assert node.label == "Deva"


# ---------------------------------------------------------------------------
# 2. Deterministic tie-break: first-seen wins on equal counts
# ---------------------------------------------------------------------------


def test_merge_tie_break_keeps_first_seen_label() -> None:
    results = [_entity_result("ANKHEG"), _entity_result("Ankheg")]
    (node,) = merge(results)[0]
    assert node.label == "ANKHEG"
    # Repeatable: same input, same result every time.
    (node_again,) = merge(results)[0]
    assert node_again.label == "ANKHEG"


def test_merge_notes_tie_break_keeps_first_seen_label() -> None:
    results = [_notes_result("abolete"), _notes_result("Abolete")]
    (node,) = merge_notes(results)[0]
    assert node.label == "abolete"
    (node_again,) = merge_notes(results)[0]
    assert node_again.label == "abolete"


# ---------------------------------------------------------------------------
# 3. Id/edge invariance: casing never affects what retrieval keys on
# ---------------------------------------------------------------------------


def test_merge_label_casing_does_not_move_node_ids_or_edges() -> None:
    base = ExtractionResult(
        entities=[
            ExtractedEntity(name="Goblin", kind="creature"),
            ExtractedEntity(name="fogo", kind="concept"),
        ],
        relations=[ExtractedRelation(subject="Goblin", predicate="RESISTS", object="fogo")],
    )
    recased = ExtractionResult(
        entities=[
            ExtractedEntity(name="GOBLIN", kind="creature"),
            ExtractedEntity(name="Fogo", kind="concept"),
        ],
        relations=[ExtractedRelation(subject="GOBLIN", predicate="RESISTS", object="Fogo")],
    )

    base_nodes, base_edges = merge([base])
    recased_nodes, recased_edges = merge([base, recased])

    base_ids = {n.id for n in base_nodes}
    recased_ids = {n.id for n in recased_nodes}
    assert recased_ids == base_ids

    base_edge_keys = {(e.src, e.dst, e.type) for e in base_edges}
    recased_edge_keys = {(e.src, e.dst, e.type) for e in recased_edges}
    assert recased_edge_keys == base_edge_keys

    seen_labels = {"Goblin", "GOBLIN", "fogo", "Fogo"}
    for node in recased_nodes:
        assert node.label in seen_labels


def test_merge_notes_label_casing_does_not_move_node_ids_or_edges() -> None:
    base = NotesExtractionResult(entities=[NotesEntity(name="Alice", kind="person")], relations=[])
    recased = NotesExtractionResult(
        entities=[NotesEntity(name="ALICE", kind="person")], relations=[]
    )

    base_nodes, base_edges = merge_notes([base])
    recased_nodes, recased_edges = merge_notes([base, recased])

    assert {n.id for n in recased_nodes} == {n.id for n in base_nodes}
    assert {(e.src, e.dst, e.type) for e in recased_edges} == {
        (e.src, e.dst, e.type) for e in base_edges
    }

    seen_labels = {"Alice", "ALICE"}
    for node in recased_nodes:
        assert node.label in seen_labels
