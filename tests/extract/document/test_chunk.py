from glyph.extract.document.chunk import Chunk, by_creature, is_heading
from glyph.extract.document.pdf import Page


def _is_caps(line: str) -> bool:
    return line.strip() == line.strip().upper() and any(c.isalpha() for c in line)


def _page(number: int, lines: list[str]) -> Page:
    """Build a Page, sizing all-caps lines like Monster Manual headings (14pt vs 9pt body)."""
    sizes = tuple(14.0 if _is_caps(line) else 9.0 for line in lines)
    return Page(
        book="mm", number=number, text="\n".join(lines), lines=tuple(lines), line_sizes=sizes
    )


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


def test_by_creature_page_without_lines_yields_nothing() -> None:
    page = Page(book="mm", number=1, text="", lines=(), line_sizes=())
    assert by_creature([page]) == []


def test_by_creature_rejects_same_size_caps_stat_labels() -> None:
    # "DUERGAR" is a heading (14pt); "FORÇA"/"AÇÕES" are all-caps but body-size (9pt/11pt).
    page = Page(
        book="mm",
        number=5,
        text="",
        lines=("DUERGAR", "FORÇA", "AÇÕES", "Os duergar vivem no Subterrâneo."),
        line_sizes=(14.0, 9.0, 11.0, 9.0),
    )
    chunks = by_creature([page])
    assert [c.label for c in chunks] == ["Duergar"]
    assert "FORÇA" in chunks[0].text
    assert "AÇÕES" in chunks[0].text


def test_by_creature_collapses_duplicate_heading_lines() -> None:
    # The Monster Manual prints each name twice (overlapping text layers).
    page = Page(
        book="mm",
        number=9,
        text="",
        lines=("DUERGAR", "DUERGAR", "corpo do verbete", "mais uma linha de corpo"),
        line_sizes=(14.0, 14.0, 9.0, 9.0),
    )
    chunks = by_creature([page])
    assert len(chunks) == 1
    assert chunks[0].label == "Duergar"
    assert chunks[0].text.count("DUERGAR") == 1
