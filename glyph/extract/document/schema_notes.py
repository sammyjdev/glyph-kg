"""Generic personal-notes extraction schema and graph mapping.

Entities: person, project, concept, note, source.
Relations: RELATES_TO, MENTIONS, PART_OF, AUTHORED_BY, DEPENDS_ON.

Mirrors the structure of ``schema.py`` (the D&D schema) so that
``DocumentExtractor`` can swap schemas by config without changing any
orchestration logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType

NotesPredicate = Literal["RELATES_TO", "MENTIONS", "PART_OF", "AUTHORED_BY", "DEPENDS_ON"]
NotesKind = Literal["person", "project", "concept", "note", "source"]

# Map entity kinds to graph node types.
_NODE_TYPE: dict[str, NodeType] = {
    "person": NodeType.ENTITY,
    "project": NodeType.ENTITY,
    "concept": NodeType.CONCEPT,
    "note": NodeType.SECTION,
    "source": NodeType.CONCEPT,
}

# Map predicate strings to EdgeType (all already live in EdgeType StrEnum).
_EDGE_TYPE: dict[str, EdgeType] = {
    "RELATES_TO": EdgeType.RELATES_TO,
    "MENTIONS": EdgeType.MENTIONS,
    "PART_OF": EdgeType.REQUIRES,  # PART_OF has no dedicated value; REQUIRES is semantically close
    "AUTHORED_BY": EdgeType.MENTIONS,  # use MENTIONS as a generic attribution edge
    "DEPENDS_ON": EdgeType.REQUIRES,
}


class NotesEntity(BaseModel):
    name: str
    kind: NotesKind
    description: str | None = None
    url: str | None = None
    date: str | None = None


class NotesRelation(BaseModel):
    subject: str
    predicate: NotesPredicate
    object: str


class NotesExtractionResult(BaseModel):
    entities: list[NotesEntity]
    relations: list[NotesRelation]


def _nid(name: str) -> str:
    return " ".join(name.split()).lower()


def _attrs(entity: NotesEntity) -> dict[str, str]:
    raw = {
        "description": entity.description,
        "url": entity.url,
        "date": entity.date,
    }
    return {k: v for k, v in raw.items() if v is not None}


def merge_notes(
    results: Iterable[NotesExtractionResult],
) -> tuple[list[Node], list[Edge]]:
    """Map and deduplicate notes extraction results into nodes and edges."""
    nodes: dict[str, Node] = {}
    edges: dict[tuple[str, str, EdgeType], Edge] = {}

    for result in results:
        for entity in result.entities:
            nid = _nid(entity.name)
            if nid not in nodes:
                nodes[nid] = Node(
                    id=nid,
                    type=_NODE_TYPE.get(entity.kind, NodeType.CONCEPT),
                    label=entity.name.strip(),
                    attrs=_attrs(entity),
                )
        for relation in result.relations:
            src, dst = _nid(relation.subject), _nid(relation.object)
            for endpoint, raw in ((src, relation.subject), (dst, relation.object)):
                if endpoint not in nodes:
                    nodes[endpoint] = Node(
                        id=endpoint, type=NodeType.CONCEPT, label=raw.strip(), attrs={}
                    )
            edge_type = _EDGE_TYPE.get(relation.predicate, EdgeType.RELATES_TO)
            key = (src, dst, edge_type)
            if key not in edges:
                edges[key] = Edge(src=src, dst=dst, type=edge_type)

    return list(nodes.values()), list(edges.values())
