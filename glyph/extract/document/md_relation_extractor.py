"""Deterministic relation extractor for Markdown documents.

Reads YAML frontmatter fields and local Markdown links to produce
REQUIRES/RELATES_TO/REFERENCES edges without any LLM call.
Satisfies the Extractor port (Source = Path | str; requires Path).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from glyph.extract.port import Source
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

_RELATION_FIELDS = frozenset(
    {"relates_to", "requires", "supersedes", "implements", "extends", "replaces"}
)

_FIELD_TO_EDGE: dict[str, EdgeType] = {
    "relates_to": EdgeType.RELATES_TO,
    "extends": EdgeType.RELATES_TO,
    "supersedes": EdgeType.RELATES_TO,
    "replaces": EdgeType.RELATES_TO,
    "requires": EdgeType.REQUIRES,
    "implements": EdgeType.REQUIRES,
}


def _frontmatter_body(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    return m.group(1) if m else ""


def _strip_frontmatter(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    return text[m.end():] if m else text


def _parse_relations(fm: str) -> list[tuple[str, str]]:
    """Return (field, target) pairs from known relation fields in YAML frontmatter."""
    relations: list[tuple[str, str]] = []
    current_field: str | None = None
    for line in fm.splitlines():
        if current_field and line.strip().startswith("- "):
            target = line.strip()[2:].strip().strip("'\"")
            if target:
                relations.append((current_field, target))
            continue
        current_field = None
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        if key not in _RELATION_FIELDS:
            continue
        val = val.strip()
        if not val:
            current_field = key
        else:
            for item in val.strip("[]").split(","):
                item = item.strip().strip("'\"")
                if item:
                    relations.append((key, item))
    return relations


def _parse_references(body: str) -> list[str]:
    """Extract local file stems from markdown links, preserving order."""
    seen: set[str] = set()
    refs: list[str] = []
    for href in _LINK_RE.findall(body):
        if href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        stem = Path(href.split("#")[0].split("?")[0]).stem
        if stem and stem not in seen:
            seen.add(stem)
            refs.append(stem)
    return refs


class MarkdownRelationExtractor:
    """Deterministic extractor: frontmatter fields + local links → graph edges.

    Requires a Path as source (reads file + derives document ID from stem).
    Zero LLM cost. Complements DocumentExtractor (LLM) which handles entity
    extraction from prose and implicit relations.
    """

    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        if not isinstance(source, Path):
            source = Path(source)
        text = source.read_text(encoding="utf-8")
        doc_id = source.stem

        fm = _frontmatter_body(text)
        body = _strip_frontmatter(text)

        relations = _parse_relations(fm)
        references = _parse_references(body)

        if not relations and not references:
            return [], []

        nodes: list[Node] = [Node(id=doc_id, type=NodeType.ENTITY, label=doc_id)]
        seen_ids: set[str] = {doc_id}
        edges: list[Edge] = []

        for field, target in relations:
            if target not in seen_ids:
                nodes.append(Node(id=target, type=NodeType.ENTITY, label=target))
                seen_ids.add(target)
            edge_type = _FIELD_TO_EDGE.get(field, EdgeType.RELATES_TO)
            attrs = {"verb": field} if field in ("supersedes", "replaces") else {}
            edges.append(Edge(src=doc_id, dst=target, type=edge_type, attrs=attrs))

        for ref in references:
            if ref == doc_id:
                continue
            if ref not in seen_ids:
                nodes.append(Node(id=ref, type=NodeType.ENTITY, label=ref))
                seen_ids.add(ref)
            edges.append(Edge(src=doc_id, dst=ref, type=EdgeType.REFERENCES))

        return nodes, edges
