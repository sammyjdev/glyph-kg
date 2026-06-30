"""S-01..S-05: MarkdownRelationExtractor deterministic relation extraction."""
import pytest
from pathlib import Path
from glyph.extract.document.md_relation_extractor import MarkdownRelationExtractor
from glyph.extract.port import Extractor
from glyph.model.edge import EdgeType
from glyph.model.node import NodeType


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# --- S-01: frontmatter relations ---

def test_s01_inline_list(tmp_path):
    p = _write(tmp_path, "dec-002.md",
        "---\nrelates_to: [dec-001, ADR-G6]\n---\n# Body\nsome text\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    edge_dsts = {e.dst for e in edges}
    assert "dec-001" in edge_dsts
    assert "ADR-G6" in edge_dsts
    assert all(e.type == EdgeType.RELATES_TO for e in edges if e.dst in ("dec-001", "ADR-G6"))


def test_s01_block_list(tmp_path):
    p = _write(tmp_path, "dec-003.md",
        "---\nsupersedes:\n  - dec-001\n  - dec-002\n---\n# Body\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    assert {e.dst for e in edges} == {"dec-001", "dec-002"}
    assert all(e.attrs.get("verb") == "supersedes" for e in edges)


def test_s01_scalar_requires(tmp_path):
    p = _write(tmp_path, "adr-007.md",
        "---\nrequires: PostgreSQL\n---\n# Body\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert any(e.dst == "PostgreSQL" and e.type == EdgeType.REQUIRES for e in edges)


def test_s01_unknown_fields_ignored(tmp_path):
    p = _write(tmp_path, "doc.md",
        "---\nid: doc\nstatus: accepted\ndate: 2026-06-30\n---\n# Body\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    assert list(nodes) == [] and list(edges) == []


# --- S-02: link references ---

def test_s02_local_links_extracted(tmp_path):
    p = _write(tmp_path, "doc.md",
        "# Body\nSee [ADR-G7](../decisions/dec-g7.md) and [dec-121](./dec-121.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    dsts = {e.dst for e in edges}
    assert "dec-g7" in dsts and "dec-121" in dsts
    assert all(e.type == EdgeType.REFERENCES for e in edges)


def test_s02_http_links_excluded(tmp_path):
    p = _write(tmp_path, "doc.md",
        "# Body\n[GitHub](https://github.com/foo) and [local](./file.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert {e.dst for e in edges} == {"file"}


def test_s02_anchor_links_excluded(tmp_path):
    p = _write(tmp_path, "doc.md",
        "# Body\n[section](#heading) and [other](./file.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert {e.dst for e in edges} == {"file"}


def test_s02_self_reference_excluded(tmp_path):
    p = _write(tmp_path, "dec-001.md",
        "# Body\n[self](./dec-001.md) and [other](./dec-002.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert all(e.dst != "dec-001" for e in edges)


# --- S-03: node shape ---

def test_s03_document_node_present(tmp_path):
    p = _write(tmp_path, "dec-005.md",
        "---\nrelates_to: [dec-004]\n---\n# Body\n")
    nodes, _ = MarkdownRelationExtractor().extract(p)
    ids = {n.id for n in nodes}
    assert "dec-005" in ids
    doc_node = next(n for n in nodes if n.id == "dec-005")
    assert doc_node.type == NodeType.ENTITY


def test_s03_target_nodes_created(tmp_path):
    p = _write(tmp_path, "dec-005.md",
        "---\nrelates_to: [dec-004]\nrequires: PostgreSQL\n---\n# Body\n")
    nodes, _ = MarkdownRelationExtractor().extract(p)
    ids = {n.id for n in nodes}
    assert "dec-004" in ids and "PostgreSQL" in ids


def test_s03_no_duplicate_nodes(tmp_path):
    p = _write(tmp_path, "dec-005.md",
        "---\nrelates_to: [dec-004]\n---\n# See [dec-004](./dec-004.md)\n")
    nodes, _ = MarkdownRelationExtractor().extract(p)
    ids = [n.id for n in nodes]
    assert len(ids) == len(set(ids))


# --- S-04: empty document ---

def test_s04_empty_returns_nothing(tmp_path):
    p = _write(tmp_path, "plain.md", "# Just a heading\nNo relations here.\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    assert list(nodes) == [] and list(edges) == []


# --- S-05: port conformance ---

def test_s05_str_source_raises_typeerror():
    with pytest.raises(TypeError, match="MarkdownRelationExtractor requires a Path"):
        MarkdownRelationExtractor().extract("some/string/path.md")


def test_s05_port(tmp_path):
    assert isinstance(MarkdownRelationExtractor(), Extractor)
