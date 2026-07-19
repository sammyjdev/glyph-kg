"""Logging on chunk-extraction failure (ruff S112 triage, issue #12).

New file — `tests/extract/document/test_extractor.py` is not modified per the
maker's no-touch rule. Covers the same `_FlakyLLM` scenario as that file's
`test_extract_skips_chunks_that_raise`, plus a `caplog` assertion that the
swallowed exception is now logged instead of silently dropped.
"""

import logging
from pathlib import Path
from typing import Any

import fitz
import pytest

from glyph.extract.document.extractor import DocumentExtractor
from glyph.extract.document.llm import Usage
from glyph.extract.document.schema import ExtractedEntity, ExtractionResult
from glyph.extract.document.schema_notes import NotesEntity, NotesExtractionResult


class _FlakyLLM:
    """Raises on the Orc chunk; succeeds on the Goblin chunk."""

    def extract(self, system: str, text: str) -> tuple[ExtractionResult, Usage]:
        if "ORC" in text:
            raise RuntimeError("simulated API failure")
        return (
            ExtractionResult(
                entities=[ExtractedEntity(name="Goblin", kind="creature")],
                relations=[],
            ),
            Usage(input_tokens=50, output_tokens=10),
        )


class _BadShapeNotesLLM:
    """Returns a schema-invalid shape on the 2nd chunk; a valid result on the 1st."""

    def __init__(self) -> None:
        self._count = 0

    def extract(self, system: str, text: str) -> tuple[Any, Usage]:
        self._count += 1
        if self._count == 2:
            # Not a NotesExtractionResult, and missing required fields, so
            # NotesExtractionResult.model_validate(...) raises.
            return {}, Usage(input_tokens=10, output_tokens=5)
        return (
            NotesExtractionResult(
                entities=[NotesEntity(name="Bob", kind="person")],
                relations=[],
            ),
            Usage(input_tokens=10, output_tokens=5),
        )


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "GOBLIN", fontsize=16)
    p1.insert_text((72, 100), "O goblin resiste a fogo.", fontsize=11)
    p1.insert_text((72, 120), "Pequeno e covarde.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 72), "ORC", fontsize=16)
    p2.insert_text((72, 100), "O orc habita cavernas.", fontsize=11)
    p2.insert_text((72, 120), "Brutal e territorial.", fontsize=11)
    doc.save(path)
    doc.close()


def test_extract_logs_a_warning_when_a_chunk_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf = tmp_path / "mm.pdf"
    _make_pdf(pdf)
    with caplog.at_level(logging.WARNING, logger="glyph.extract.document.extractor"):
        nodes, _edges, usages = DocumentExtractor(llm=_FlakyLLM()).extract_with_usage(pdf)

    assert len(usages) == 1  # the failing Orc chunk is skipped, not fatal
    assert any(n.id == "goblin" for n in nodes)
    assert any(
        record.levelno == logging.WARNING and "Orc" in record.getMessage()
        for record in caplog.records
    )


def _write_md(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_extract_logs_a_warning_when_notes_coercion_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Covers the notes-domain coercion except-block (raw_result not already a
    NotesExtractionResult and model_validate rejects it), distinct from the
    LLM-extraction-failure path covered above.
    """
    md = _write_md(tmp_path, "note.md", "# Section A\nbody\n# Section B\nbody\n")
    with caplog.at_level(logging.WARNING, logger="glyph.extract.document.extractor"):
        nodes, _edges, usages = DocumentExtractor(
            llm=_BadShapeNotesLLM(), domain="notes"
        ).extract_with_usage(md)

    assert len(usages) == 1  # the bad-shape chunk is skipped, not fatal
    assert any(n.id == "bob" for n in nodes)
    assert any(
        record.levelno == logging.WARNING and "coercion" in record.getMessage()
        for record in caplog.records
    )
