"""LLM extraction contract and the pure mapping to the graph model.

Structured outputs forbid open dicts (``additionalProperties`` must be false), so
entity attributes are fixed optional fields rather than a free-form map.
"""

from collections import Counter
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType

Predicate = Literal["RESISTS", "IMMUNE_TO", "VULNERABLE_TO", "INHABITS", "SUMMONS"]
Kind = Literal["creature", "concept"]


class ExtractedEntity(BaseModel):
    name: str
    kind: Kind
    challenge_rating: str | None = None
    creature_type: str | None = None
    alignment: str | None = None


class ExtractedRelation(BaseModel):
    subject: str
    predicate: Predicate
    object: str


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


_NODE_TYPE: dict[str, NodeType] = {
    "creature": NodeType.ENTITY,
    "concept": NodeType.CONCEPT,
}


def _nid(name: str) -> str:
    return " ".join(name.split()).lower()


def _attrs(entity: ExtractedEntity) -> dict[str, str]:
    fields = {
        "challenge_rating": entity.challenge_rating,
        "creature_type": entity.creature_type,
        "alignment": entity.alignment,
    }
    return {k: v for k, v in fields.items() if v is not None}


def merge(results: Iterable[ExtractionResult]) -> tuple[list[Node], list[Edge]]:
    """Map and deduplicate extraction results into nodes and edges.

    A node's label is the most-frequent surface form seen for its id across all
    results (ties break to the first-seen form, since ``Counter.most_common``
    preserves insertion order for equal counts). ``id``/``type``/``attrs`` keep
    the existing first-seen semantics.
    """
    nodes: dict[str, Node] = {}
    edges: dict[tuple[str, str, EdgeType], Edge] = {}
    label_counts: dict[str, Counter[str]] = {}
    for result in results:
        for entity in result.entities:
            nid = _nid(entity.name)
            label_counts.setdefault(nid, Counter())[entity.name.strip()] += 1
            if nid not in nodes:
                nodes[nid] = Node(
                    id=nid,
                    type=_NODE_TYPE[entity.kind],
                    label=entity.name.strip(),
                    attrs=_attrs(entity),
                )
        for relation in result.relations:
            src, dst = _nid(relation.subject), _nid(relation.object)
            for endpoint, raw in ((src, relation.subject), (dst, relation.object)):
                label_counts.setdefault(endpoint, Counter())[raw.strip()] += 1
                if endpoint not in nodes:
                    nodes[endpoint] = Node(
                        id=endpoint, type=NodeType.CONCEPT, label=raw.strip(), attrs={}
                    )
            edge_type = EdgeType[relation.predicate]
            key = (src, dst, edge_type)
            if key not in edges:
                edges[key] = Edge(src=src, dst=dst, type=edge_type)
    for nid in list(nodes):
        best_label = label_counts[nid].most_common(1)[0][0]
        if best_label != nodes[nid].label:
            nodes[nid] = nodes[nid].model_copy(update={"label": best_label})
    return list(nodes.values()), list(edges.values())
