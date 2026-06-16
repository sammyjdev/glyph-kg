"""Tests for the ``by_heading`` chunker (Markdown-aware splitting)."""

from __future__ import annotations

from glyph.extract.document.chunk import Chunk, by_heading
from glyph.extract.document.pdf import Page


def _page(lines: list[str], sizes: list[float], book: str = "vault", number: int = 1) -> Page:
    return Page(
        book=book,
        number=number,
        text="\n".join(lines),
        lines=tuple(lines),
        line_sizes=tuple(sizes),
    )


def _md_page(content: str, book: str = "vault") -> Page:
    """Build a Page as the markdown loader would for a single note."""
    import pathlib
    import tempfile

    from glyph.extract.document.markdown import load

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
        f.write(content)
        tmp = pathlib.Path(f.name)
    pages = load(tmp, book=book)
    tmp.unlink()
    return pages[0]


# ---------------------------------------------------------------------------
# Core splitting behaviour
# ---------------------------------------------------------------------------


def test_by_heading_splits_at_nonzero_size_lines() -> None:
    page = _page(
        lines=["# Section A", "body a", "# Section B", "body b"],
        sizes=[6.0, 0.0, 6.0, 0.0],
    )
    chunks = by_heading([page])
    assert len(chunks) == 2
    assert "Section A" in chunks[0].label
    assert "Section B" in chunks[1].label


def test_by_heading_first_chunk_label_is_heading_text() -> None:
    page = _page(
        lines=["# My Note", "some content"],
        sizes=[6.0, 0.0],
    )
    (chunk,) = by_heading([page])
    assert chunk.label == "My Note"


def test_by_heading_body_text_included_in_chunk() -> None:
    page = _page(
        lines=["# Topic", "line one", "line two"],
        sizes=[6.0, 0.0, 0.0],
    )
    (chunk,) = by_heading([page])
    assert "line one" in chunk.text
    assert "line two" in chunk.text


def test_by_heading_no_heading_produces_one_chunk() -> None:
    page = _page(
        lines=["just body", "more body"],
        sizes=[0.0, 0.0],
    )
    chunks = by_heading([page])
    assert len(chunks) == 1
    assert chunks[0].label is not None  # falls back to None label → "p.1" via _make


def test_by_heading_empty_pages_returns_empty() -> None:
    assert by_heading([]) == []


def test_by_heading_preserves_book_name() -> None:
    page = _page(["# H", "body"], [6.0, 0.0], book="my-vault")
    (chunk,) = by_heading([page])
    assert chunk.book == "my-vault"


def test_by_heading_chunk_pages_contains_page_number() -> None:
    page = _page(["# H", "body"], [6.0, 0.0], number=3)
    (chunk,) = by_heading([page])
    assert 3 in chunk.pages


def test_by_heading_consecutive_headings_each_get_chunk() -> None:
    page = _page(
        lines=["# A", "# B", "body"],
        sizes=[6.0, 6.0, 0.0],
    )
    chunks = by_heading([page])
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Integration: markdown loader → by_heading
# ---------------------------------------------------------------------------


def test_by_heading_on_markdown_loaded_page() -> None:
    content = "# Introduction\nsome intro text\n## Sub-section\ndetails here\n"
    page = _md_page(content)
    chunks = by_heading([page])
    assert len(chunks) == 2
    assert chunks[0].label == "Introduction"
    assert "Sub-section" in chunks[1].label


def test_by_heading_produces_chunk_instances() -> None:
    page = _page(["# H", "body"], [6.0, 0.0])
    (chunk,) = by_heading([page])
    assert isinstance(chunk, Chunk)
