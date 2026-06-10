"""P1.2: split pages into per-creature chunks by ALL-CAPS heading detection.

Headings are detected by case (the Monster Manual prints creature names in caps
and has no usable PDF outline). Content before the first heading is emitted as a
page-labelled fallback chunk. Font size is available from the PDF if caps-based
detection ever proves insufficient.
"""

from dataclasses import dataclass

from glyph.extract.document.pdf import Page


@dataclass(frozen=True)
class Chunk:
    """A unit of text sent to the extractor — ideally one creature entry."""

    label: str
    text: str
    book: str
    pages: tuple[int, ...]


def is_heading(line: str) -> bool:
    """True when ``line`` looks like a creature heading: short and all-caps."""
    text = line.strip()
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2:
        return False
    if text != text.upper():
        return False
    return len(text.split()) <= 6


def _make(label: str | None, lines: list[str], nums: list[int], book: str) -> Chunk:
    resolved = label if label is not None else f"p.{nums[0]}"
    return Chunk(label=resolved, text="\n".join(lines), book=book, pages=tuple(sorted(set(nums))))


def by_creature(pages: list[Page]) -> list[Chunk]:
    """Group lines into chunks, starting a new chunk at each heading."""
    if not pages:
        return []
    book = pages[0].book
    chunks: list[Chunk] = []
    label: str | None = None
    lines: list[str] = []
    nums: list[int] = []
    for page in pages:
        for line in page.lines:
            if is_heading(line):
                if lines:
                    chunks.append(_make(label, lines, nums, book))
                label = line.strip().title()
                lines = [line]
                nums = [page.number]
            else:
                lines.append(line)
                nums.append(page.number)
    if lines:
        chunks.append(_make(label, lines, nums, book))
    return chunks
