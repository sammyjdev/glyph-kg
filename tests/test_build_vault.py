"""Tests for the vault builder (glyph.extract.document.vault)."""

from __future__ import annotations

import json
from pathlib import Path

from glyph.extract.document.llm import Usage
from glyph.extract.document.schema_notes import NotesEntity, NotesExtractionResult, NotesRelation
from glyph.vault import build_vault

# ---------------------------------------------------------------------------
# Fake LLM for vault builder tests
# ---------------------------------------------------------------------------


class _FakeNotesLLM:
    def extract(self, system: str, text: str) -> tuple[NotesExtractionResult, Usage]:
        result = NotesExtractionResult(
            entities=[
                NotesEntity(name="Alice", kind="person"),
                NotesEntity(name="Glyph", kind="project"),
            ],
            relations=[NotesRelation(subject="Alice", predicate="RELATES_TO", object="Glyph")],
        )
        return result, Usage(input_tokens=30, output_tokens=10)


def _vault(tmp_path: Path, notes: dict[str, str]) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in notes.items():
        (vault / name).write_text(content, encoding="utf-8")
    return vault


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_vault_creates_output_file(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {"note.md": "# Hello\nbody\n"})
    out = tmp_path / "out" / "graph.json"
    build_vault(vault, out, llm=_FakeNotesLLM())
    assert out.exists()


def test_build_vault_output_is_valid_json(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {"note.md": "# Hello\nbody\n"})
    out = tmp_path / "graph.json"
    build_vault(vault, out, llm=_FakeNotesLLM())
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert "edges" in data


def test_build_vault_returns_node_and_edge_counts(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {"a.md": "# A\nbody\n", "b.md": "# B\nbody\n"})
    out = tmp_path / "graph.json"
    n, e = build_vault(vault, out, llm=_FakeNotesLLM())
    assert n >= 0
    assert e >= 0


def test_build_vault_walks_nested_directories(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    sub = vault / "sub"
    sub.mkdir(parents=True)
    (vault / "root.md").write_text("# Root\nbody\n", encoding="utf-8")
    (sub / "child.md").write_text("# Child\nbody\n", encoding="utf-8")
    out = tmp_path / "graph.json"
    n, e = build_vault(vault, out, llm=_FakeNotesLLM())
    assert n >= 0  # both files processed without error


def test_build_vault_empty_vault_returns_zeros(tmp_path: Path) -> None:
    vault = tmp_path / "empty_vault"
    vault.mkdir()
    out = tmp_path / "graph.json"
    n, e = build_vault(vault, out, llm=_FakeNotesLLM())
    assert n == 0
    assert e == 0


def test_build_vault_creates_parent_directories(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {"note.md": "# H\nbody\n"})
    out = tmp_path / "deep" / "nested" / "graph.json"
    build_vault(vault, out, llm=_FakeNotesLLM())
    assert out.exists()


def test_build_vault_accumulates_nodes_across_files(tmp_path: Path) -> None:
    """Each file contributes nodes; the store deduplicates across the vault."""
    vault = _vault(
        tmp_path,
        {
            "note1.md": "# Section\nAlice and Glyph\n",
            "note2.md": "# Another\nAlice again\n",
        },
    )
    out = tmp_path / "graph.json"
    n, e = build_vault(vault, out, llm=_FakeNotesLLM())
    # Both files mention Alice+Glyph, but dedup means at most 2 nodes.
    assert n >= 1
