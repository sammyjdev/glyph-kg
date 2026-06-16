"""Tests for the Markdown/Obsidian page loader."""

from __future__ import annotations

from pathlib import Path

from glyph.extract.document.markdown import load
from glyph.extract.document.pdf import Page


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------


def test_load_returns_single_page(tmp_path: Path) -> None:
    f = _write(tmp_path, "note.md", "# Hello\nsome body\n")
    pages = load(f)
    assert len(pages) == 1
    assert isinstance(pages[0], Page)


def test_load_book_defaults_to_stem(tmp_path: Path) -> None:
    f = _write(tmp_path, "my-note.md", "body\n")
    pages = load(f)
    assert pages[0].book == "my-note"


def test_load_book_override(tmp_path: Path) -> None:
    f = _write(tmp_path, "note.md", "body\n")
    pages = load(f, book="vault")
    assert pages[0].book == "vault"


def test_load_page_number_is_one(tmp_path: Path) -> None:
    f = _write(tmp_path, "note.md", "body\n")
    pages = load(f)
    assert pages[0].number == 1


# ---------------------------------------------------------------------------
# Heading detection via line_sizes
# ---------------------------------------------------------------------------


def test_h1_heading_has_nonzero_size(tmp_path: Path) -> None:
    f = _write(tmp_path, "note.md", "# Title\nbody\n")
    page = load(f)[0]
    # First line is the heading
    assert page.line_sizes[0] > 0.0


def test_body_line_has_zero_size(tmp_path: Path) -> None:
    f = _write(tmp_path, "note.md", "# Title\njust body text\n")
    page = load(f)[0]
    assert page.line_sizes[1] == 0.0


def test_deeper_headings_have_smaller_sizes(tmp_path: Path) -> None:
    content = "# H1\n## H2\n### H3\nbody\n"
    f = _write(tmp_path, "note.md", content)
    page = load(f)[0]
    sizes = page.line_sizes
    # H1 > H2 > H3 > body (0.0)
    assert sizes[0] > sizes[1] > sizes[2] > sizes[3]


def test_lines_and_sizes_are_parallel(tmp_path: Path) -> None:
    content = "# Section\nfirst\nsecond\n"
    f = _write(tmp_path, "note.md", content)
    page = load(f)[0]
    assert len(page.lines) == len(page.line_sizes)


# ---------------------------------------------------------------------------
# YAML frontmatter stripping
# ---------------------------------------------------------------------------


def test_frontmatter_is_stripped(tmp_path: Path) -> None:
    content = "---\ntitle: My Note\ntags: [python]\n---\n# Real content\nbody\n"
    f = _write(tmp_path, "note.md", content)
    page = load(f)[0]
    assert not any("title:" in line for line in page.lines)
    assert any("Real content" in line for line in page.lines)


def test_no_frontmatter_is_fine(tmp_path: Path) -> None:
    content = "# Title\nbody\n"
    f = _write(tmp_path, "note.md", content)
    page = load(f)[0]
    assert page.lines[0] == "# Title"


def test_frontmatter_headings_not_counted_as_content_headings(tmp_path: Path) -> None:
    """Lines inside frontmatter must not appear as heading lines."""
    content = "---\ntitle: Ignored\n---\n# Real heading\nbody\n"
    f = _write(tmp_path, "note.md", content)
    page = load(f)[0]
    # The first line should be the real heading, not "title: Ignored".
    assert "Real heading" in page.lines[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(tmp_path: Path) -> None:
    f = _write(tmp_path, "empty.md", "")
    pages = load(f)
    assert len(pages) == 1
    assert pages[0].lines == ()


def test_trailing_blank_lines_stripped(tmp_path: Path) -> None:
    content = "# Title\nbody\n\n\n"
    f = _write(tmp_path, "note.md", content)
    page = load(f)[0]
    assert page.lines[-1] != ""


def test_str_path_accepted(tmp_path: Path) -> None:
    f = _write(tmp_path, "note.md", "body\n")
    pages = load(str(f))
    assert len(pages) == 1


def test_text_field_contains_full_body(tmp_path: Path) -> None:
    content = "---\nkey: val\n---\n# H\nbody\n"
    f = _write(tmp_path, "note.md", content)
    page = load(f)[0]
    assert "# H" in page.text
    assert "body" in page.text
    assert "key: val" not in page.text
