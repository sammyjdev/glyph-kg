from pathlib import Path

import fitz

from glyph.extract.document.pdf import Page, load


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "GOBLIN\nO goblin resiste a fogo.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 72), "ORC\nO orc habita cavernas.", fontsize=11)
    doc.save(path)
    doc.close()


def test_load_returns_one_page_per_pdf_page(tmp_path: Path) -> None:
    pdf = tmp_path / "bestiary.pdf"
    _make_pdf(pdf)
    pages = load(pdf)
    assert len(pages) == 2
    assert all(isinstance(p, Page) for p in pages)


def test_load_carries_book_and_one_based_page_number(tmp_path: Path) -> None:
    pdf = tmp_path / "bestiary.pdf"
    _make_pdf(pdf)
    pages = load(pdf, book="bestiary")
    assert pages[0].book == "bestiary"
    assert pages[0].number == 1
    assert pages[1].number == 2


def test_load_defaults_book_to_file_stem(tmp_path: Path) -> None:
    pdf = tmp_path / "Monster Manual.pdf"
    _make_pdf(pdf)
    pages = load(pdf)
    assert pages[0].book == "Monster Manual"


def test_load_extracts_text_and_lines(tmp_path: Path) -> None:
    pdf = tmp_path / "bestiary.pdf"
    _make_pdf(pdf)
    pages = load(pdf)
    assert "GOBLIN" in pages[0].text
    assert "GOBLIN" in pages[0].lines
    assert "O goblin resiste a fogo." in pages[0].lines


def test_load_captures_a_font_size_per_line(tmp_path: Path) -> None:
    pdf = tmp_path / "sized.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "GOBLIN", fontsize=16)
    page.insert_text((72, 100), "corpo do verbete", fontsize=11)
    doc.save(pdf)
    doc.close()
    pages = load(pdf)
    assert len(pages[0].line_sizes) == len(pages[0].lines)
    assert all(isinstance(s, float) for s in pages[0].line_sizes)
    sizes = dict(zip(pages[0].lines, pages[0].line_sizes, strict=True))
    assert sizes["GOBLIN"] > sizes["corpo do verbete"]


def test_load_drops_blank_lines(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "GOBLIN\n   \nfim", fontsize=11)
    doc.save(pdf)
    doc.close()
    pages = load(pdf)
    assert "GOBLIN" in pages[0].lines
    assert "fim" in pages[0].lines
    assert "" not in pages[0].lines
    assert all(line.strip() for line in pages[0].lines)
