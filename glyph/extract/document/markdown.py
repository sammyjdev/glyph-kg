"""P2.1: read a Markdown / Obsidian note into pages with text and per-line metadata.

YAML frontmatter (``---`` … ``---``) is stripped before processing.  Heading
lines (``# …``, ``## …``, …) carry a non-zero ``line_size`` so that
:func:`glyph.extract.document.chunk.by_heading` can split on them; body lines
carry ``0.0``.  The heading level maps to a descending size sequence
(``# = 6.0``, ``## = 5.0``, …) mirroring the ``line_sizes`` semantics of
:func:`glyph.extract.document.pdf.load` without importing PyMuPDF.

A Markdown file is represented as a single "page" (``number=1``) because the
concept of page-numbers does not apply; consumers that need sections can use the
chunk label instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from glyph.extract.document.pdf import Page
from glyph.extract.port import Source

# Heading-level → proxy font size (larger = higher-level heading).
_HEADING_SIZE: dict[int, float] = {1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0, 5: 2.0, 6: 1.0}

_HEADING_RE = re.compile(r"^(#{1,6})\s")
_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block if present."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def load(source: Source, book: str | None = None) -> list[Page]:
    """Read a single Markdown file into a one-element list of :class:`Page`.

    Parameters
    ----------
    source:
        Path to the ``.md`` / ``.markdown`` file (or a ``str`` path).
    book:
        Optional logical book/vault name; defaults to the file stem.

    Returns
    -------
    list[Page]
        A single ``Page`` whose ``lines``/``line_sizes`` encode heading vs. body
        so that :func:`~glyph.extract.document.chunk.by_heading` can split it.
    """
    path = Path(source)
    book_name = book if book is not None else path.stem
    raw = path.read_text(encoding="utf-8")
    body = _strip_frontmatter(raw)

    lines: list[str] = []
    sizes: list[float] = []

    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            lines.append(line)
            sizes.append(_HEADING_SIZE.get(level, 1.0))
        else:
            lines.append(line)
            sizes.append(0.0)

    # Drop trailing blank lines to avoid an empty trailing chunk.
    while lines and not lines[-1].strip():
        lines.pop()
        sizes.pop()

    return [
        Page(
            book=book_name,
            number=1,
            text=body,
            lines=tuple(lines),
            line_sizes=tuple(sizes),
        )
    ]
