"""P1.1: read a PDF into pages with text, per-line text, and metadata."""

from dataclasses import dataclass
from pathlib import Path

import fitz

from glyph.extract.port import Source


@dataclass(frozen=True)
class Page:
    """One PDF page: full text, its lines, and each line's font size.

    ``lines`` and ``line_sizes`` are parallel: ``line_sizes[i]`` is the largest
    span font size on ``lines[i]``. Size separates creature-name headings (large
    font) from same-page stat-block labels (body font) during chunking.
    """

    book: str
    number: int  # 1-based
    text: str
    lines: tuple[str, ...]
    line_sizes: tuple[float, ...]


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
            sizes: list[float] = []
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span["text"] for span in spans).strip()
                    if text:
                        lines.append(text)
                        sizes.append(max((span["size"] for span in spans), default=0.0))
            pages.append(
                Page(
                    book=book_name,
                    number=index + 1,
                    text=page.get_text("text"),
                    lines=tuple(lines),
                    line_sizes=tuple(sizes),
                )
            )
        return pages
    finally:
        doc.close()
