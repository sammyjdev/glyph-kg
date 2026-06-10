from glyph.extract.document.chunk import Chunk, by_creature, is_heading
from glyph.extract.document.pdf import Page


def _page(number: int, lines: list[str]) -> Page:
    return Page(book="mm", number=number, text="\n".join(lines), lines=tuple(lines))


def test_is_heading_accepts_all_caps_short_lines() -> None:
    assert is_heading("GOBLIN")
    assert is_heading("ABOCANHADOR MATRAQUEANTE")
    assert is_heading("KUO-TOA")


def test_is_heading_rejects_body_and_page_numbers() -> None:
    assert not is_heading("O goblin resiste a fogo.")
    assert not is_heading("13")
    assert not is_heading("")


def test_by_creature_splits_on_headings() -> None:
    pages = [
        _page(1, ["GOBLIN", "O goblin resiste a fogo.", "ORC", "O orc habita cavernas."]),
    ]
    chunks = by_creature(pages)
    assert [c.label for c in chunks] == ["Goblin", "Orc"]
    assert "resiste a fogo" in chunks[0].text
    assert "habita cavernas" in chunks[1].text


def test_by_creature_continues_a_creature_across_pages() -> None:
    pages = [
        _page(1, ["GOBLIN", "Primeira parte."]),
        _page(2, ["Continuacao na pagina dois."]),
    ]
    chunks = by_creature(pages)
    assert len(chunks) == 1
    assert chunks[0].pages == (1, 2)
    assert "Continuacao" in chunks[0].text


def test_by_creature_labels_pre_heading_content_by_page() -> None:
    pages = [_page(1, ["Texto introdutorio sem cabecalho.", "Mais texto."])]
    chunks = by_creature(pages)
    assert len(chunks) == 1
    assert chunks[0].label == "p.1"


def test_chunk_is_a_dataclass_with_book() -> None:
    pages = [_page(7, ["GOBLIN", "corpo"])]
    (chunk,) = by_creature(pages)
    assert isinstance(chunk, Chunk)
    assert chunk.book == "mm"


def test_by_creature_empty_pages_returns_empty() -> None:
    assert by_creature([]) == []
