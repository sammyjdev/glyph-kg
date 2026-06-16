"""Tests for DocumentExtractor markdown dispatch and notes domain."""

from __future__ import annotations

from pathlib import Path

from glyph.extract.document.extractor import DocumentExtractor
from glyph.extract.document.llm import Usage
from glyph.extract.document.schema_notes import NotesEntity, NotesExtractionResult, NotesRelation

# ---------------------------------------------------------------------------
# Fake LLM returning notes-domain JSON
# ---------------------------------------------------------------------------


class _FakeNotesLLM:
    """Returns a NotesExtractionResult regardless of system/text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, system: str, text: str) -> tuple[NotesExtractionResult, Usage]:
        self.calls.append(text)
        result = NotesExtractionResult(
            entities=[
                NotesEntity(name="Alice", kind="person"),
                NotesEntity(name="Glyph", kind="project"),
            ],
            relations=[NotesRelation(subject="Alice", predicate="RELATES_TO", object="Glyph")],
        )
        return result, Usage(input_tokens=50, output_tokens=10)


class _FlakyNotesLLM:
    """Raises on the second call."""

    def __init__(self) -> None:
        self._count = 0

    def extract(self, system: str, text: str) -> tuple[NotesExtractionResult, Usage]:
        self._count += 1
        if self._count == 2:
            raise RuntimeError("simulated failure")
        return (
            NotesExtractionResult(
                entities=[NotesEntity(name="Bob", kind="person")],
                relations=[],
            ),
            Usage(input_tokens=10, output_tokens=5),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_md(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Markdown dispatch tests
# ---------------------------------------------------------------------------


def test_extractor_dispatches_md_extension(tmp_path: Path) -> None:
    md = _write_md(tmp_path, "note.md", "# Section\nsome content\n")
    llm = _FakeNotesLLM()
    extractor = DocumentExtractor(llm=llm, domain="notes")
    nodes, edges, usages = extractor.extract_with_usage(md)
    assert llm.calls  # LLM was called at least once
    assert nodes
    assert edges


def test_extractor_dispatches_markdown_extension(tmp_path: Path) -> None:
    md = _write_md(tmp_path, "note.markdown", "# Section\nsome content\n")
    llm = _FakeNotesLLM()
    extractor = DocumentExtractor(llm=llm, domain="notes")
    nodes, edges, _ = extractor.extract_with_usage(md)
    assert nodes


def test_extractor_notes_domain_produces_nodes(tmp_path: Path) -> None:
    md = _write_md(tmp_path, "note.md", "# Intro\nAlice worked on Glyph.\n")
    nodes, edges, usages = DocumentExtractor(
        llm=_FakeNotesLLM(), domain="notes"
    ).extract_with_usage(md)
    by_id = {n.id: n for n in nodes}
    assert "alice" in by_id
    assert "glyph" in by_id
    assert len(usages) >= 1


def test_extractor_notes_domain_produces_edges(tmp_path: Path) -> None:
    md = _write_md(tmp_path, "note.md", "# Intro\nAlice worked on Glyph.\n")
    _, edges, _ = DocumentExtractor(llm=_FakeNotesLLM(), domain="notes").extract_with_usage(md)
    assert edges


def test_extractor_notes_skips_failed_chunks(tmp_path: Path) -> None:
    content = "# Section A\nbody\n# Section B\nbody\n"
    md = _write_md(tmp_path, "note.md", content)
    nodes, _, usages = DocumentExtractor(llm=_FlakyNotesLLM(), domain="notes").extract_with_usage(
        md
    )
    # One chunk raises; we still get results from the other.
    assert len(usages) == 1
    assert any(n.id == "bob" for n in nodes)


def test_extractor_extract_public_api_works_with_markdown(tmp_path: Path) -> None:
    md = _write_md(tmp_path, "note.md", "# H\nbody\n")
    extractor = DocumentExtractor(llm=_FakeNotesLLM(), domain="notes")
    nodes, edges = extractor.extract(md)
    assert nodes is not None
    assert edges is not None


def test_extractor_notes_system_prompt_passed_to_llm(tmp_path: Path) -> None:
    """The system passed to the LLM must contain notes-domain predicates."""
    calls: list[str] = []

    class _CaptureLLM:
        def extract(self, system: str, text: str) -> tuple[NotesExtractionResult, Usage]:
            calls.append(system)
            return NotesExtractionResult(entities=[], relations=[]), Usage(0, 0)

    md = _write_md(tmp_path, "note.md", "# H\nbody\n")
    DocumentExtractor(llm=_CaptureLLM(), domain="notes").extract_with_usage(md)
    assert calls
    assert "RELATES_TO" in calls[0] or "MENTIONS" in calls[0]


def test_extractor_dnd_domain_unchanged(tmp_path: Path) -> None:
    """Regression: default dnd domain with a real fake PDF still works.

    We only test that no exception is raised when domain='dnd' is passed
    explicitly; the full PDF path is tested in test_extractor.py.
    """
    # We can't easily create a PDF here without fitz; just verify the
    # extractor initializes correctly with domain='dnd'.
    extractor = DocumentExtractor(domain="dnd")
    assert extractor._domain == "dnd"
