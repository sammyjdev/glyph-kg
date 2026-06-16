"""DocumentExtractor: orchestrate document -> chunk -> LLM -> graph, behind the Extractor port.

Supported source types (dispatched by file extension):
- ``.pdf``              → PyMuPDF loader + ``by_creature`` chunking (D&D default)
- ``.md`` / ``.markdown`` → Markdown loader + ``by_heading`` chunking (generic notes)

Domain is selected via the ``domain`` constructor argument (``"dnd"`` or ``"notes"``).
The LLM extractor and keep-filter are still injectable for testing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from glyph.extract.document import chunk as chunking
from glyph.extract.document import markdown, pdf
from glyph.extract.document.llm import AnthropicExtractor, LLMExtractor, Usage
from glyph.extract.document.prompt import get_system_prompt
from glyph.extract.document.schema import ExtractionResult, merge
from glyph.extract.document.schema_notes import NotesExtractionResult, merge_notes
from glyph.extract.port import Source
from glyph.model.edge import Edge
from glyph.model.node import Node


class DocumentExtractor:
    """Probabilistic extractor: reads a document and infers entities/relations with an LLM.

    Parameters
    ----------
    llm:
        An object satisfying the ``LLMExtractor`` Protocol.  Defaults to
        ``AnthropicExtractor`` (Haiku 4.5) when ``None``.
    keep:
        Optional filter applied to chunks before LLM calls.  ``None`` keeps all.
    domain:
        ``"dnd"``   → D&D schema + ``by_creature`` chunking (default).
        ``"notes"`` → generic notes schema + ``by_heading`` chunking.
    """

    def __init__(
        self,
        llm: LLMExtractor | None = None,
        keep: Callable[[chunking.Chunk], bool] | None = None,
        domain: str = "dnd",
    ) -> None:
        self._llm = llm if llm is not None else AnthropicExtractor()
        self._keep = keep
        self._domain = domain

    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        nodes, edges, _ = self.extract_with_usage(source)
        return nodes, edges

    def extract_with_usage(self, source: Source) -> tuple[list[Node], list[Edge], list[Usage]]:
        path = Path(source)
        suffix = path.suffix.lower()

        system = get_system_prompt(self._domain)

        # --- load & chunk --------------------------------------------------
        if suffix in {".md", ".markdown"}:
            pages = markdown.load(source)
            chunks = chunking.by_heading(pages)
        else:
            # Default: treat as PDF (preserves existing behaviour for .pdf and
            # any other extension that was previously accepted).
            pages = pdf.load(source)
            chunks = chunking.by_creature(pages)

        if self._keep is not None:
            chunks = [piece for piece in chunks if self._keep(piece)]

        # --- LLM extraction ------------------------------------------------
        results_dnd: list[ExtractionResult] = []
        results_notes: list[NotesExtractionResult] = []
        usages: list[Usage] = []

        for piece in chunks:
            try:
                raw_result, usage = self._llm.extract(system, piece.text)
            except Exception:  # noqa: BLE001 - one failed chunk must not abort a paid run
                continue
            if self._domain == "notes":
                if isinstance(raw_result, NotesExtractionResult):
                    results_notes.append(raw_result)
                else:
                    # LLM returned a dict/object; try coercion
                    try:
                        results_notes.append(
                            NotesExtractionResult.model_validate(
                                raw_result.model_dump()
                                if hasattr(raw_result, "model_dump")
                                else raw_result
                            )
                        )
                    except Exception:  # noqa: BLE001
                        continue
            else:
                results_dnd.append(raw_result)
            usages.append(usage)

        # --- merge ---------------------------------------------------------
        if self._domain == "notes":
            nodes, edges = merge_notes(results_notes)
        else:
            nodes, edges = merge(results_dnd)

        return nodes, edges, usages
