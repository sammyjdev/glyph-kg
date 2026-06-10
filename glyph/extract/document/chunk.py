"""P1.2: split pages into per-creature chunks by heading detection.

A heading is an ALL-CAPS line in a font clearly larger than the body. The case
rule alone over-segments the Monster Manual: stat-block labels (``FORÇA``,
``AÇÕES``, ``CR 3``) are also all-caps but print in the body font, so the font
size gate keeps only the creature names. Creature names are also printed twice
(overlapping text layers); consecutive headings with no body between them are
collapsed into one chunk. Content before the first heading is a page-labelled
fallback chunk.
"""

from collections import Counter
from dataclasses import dataclass

from glyph.extract.document.pdf import Page

# A heading's font must exceed the body font by this factor.
HEADING_SIZE_FACTOR = 1.3


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


def _body_size(pages: list[Page]) -> float:
    """The body font: the most common line size, breaking ties toward the smaller.

    Frequency ignores rare large outliers (chapter titles); the smaller-on-tie rule
    resolves a prose/label split (e.g. 9.0 vs 9.5) toward the true body size.
    """
    sizes = [size for page in pages for size in page.line_sizes]
    if not sizes:
        return 0.0
    counts = Counter(sizes)
    most_common = max(counts.values())
    return min(size for size, count in counts.items() if count == most_common)


def by_creature(pages: list[Page]) -> list[Chunk]:
    """Group lines into chunks, starting a new chunk at each creature heading."""
    if not pages:
        return []
    book = pages[0].book
    threshold = _body_size(pages) * HEADING_SIZE_FACTOR
    chunks: list[Chunk] = []
    label: str | None = None
    lines: list[str] = []
    nums: list[int] = []
    has_body = False
    for page in pages:
        for line, size in zip(page.lines, page.line_sizes, strict=True):
            if is_heading(line) and size >= threshold:
                if has_body:
                    chunks.append(_make(label, lines, nums, book))
                    label, lines, nums, has_body = None, [], [], False
                if label is None:  # first heading of the block; duplicates are skipped
                    label = line.strip().title()
                    lines.append(line)
                    nums.append(page.number)
            else:
                lines.append(line)
                nums.append(page.number)
                has_body = True
    if lines:
        chunks.append(_make(label, lines, nums, book))
    return chunks
