"""P1.1: read a PDF into pages with text, per-line text, and metadata."""

from dataclasses import dataclass
from pathlib import Path

import fitz

from glyph.extract.port import Source


@dataclass(frozen=True)
class Page:
    """One PDF page: full text plus the individual lines (for heading detection)."""

    book: str
    number: int  # 1-based
    text: str
    lines: tuple[str, ...]


def load(source: Source, book: str | None = None) -> list[Page]:
    """Read every page of ``source`` into :class:`Page` objects."""
    path = Path(source)
    book_name = book if book is not None else path.stem
    doc = fitz.open(path)
    try:
        pages: list[Page] = []
        for index in range(doc.page_count):
            page = doc[index]
            data = page.get_text("dict")
            lines: list[str] = []
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line.get("spans", [])).strip()
                    if text:
                        lines.append(text)
            pages.append(
                Page(
                    book=book_name,
                    number=index + 1,
                    text=page.get_text("text"),
                    lines=tuple(lines),
                )
            )
        return pages
    finally:
        doc.close()
