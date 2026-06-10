from pathlib import Path

import fitz

from glyph.extract.document.extractor import DocumentExtractor
from glyph.extract.document.llm import Usage
from glyph.extract.document.schema import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from glyph.extract.port import Extractor
from glyph.model.edge import EdgeType
from glyph.model.node import NodeType


class _FakeLLM:
    """Returns one extraction per chunk, keyed by the creature heading in the text."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def extract(self, system: str, text: str) -> tuple[ExtractionResult, Usage]:
        self.seen.append(text)
        if "GOBLIN" in text:
            result = ExtractionResult(
                entities=[
                    ExtractedEntity(name="Goblin", kind="creature"),
                    ExtractedEntity(name="fogo", kind="concept"),
                ],
                relations=[ExtractedRelation(subject="Goblin", predicate="RESISTS", object="fogo")],
            )
        else:
            result = ExtractionResult(
                entities=[ExtractedEntity(name="Orc", kind="creature")],
                relations=[
                    ExtractedRelation(subject="Orc", predicate="INHABITS", object="cavernas")
                ],
            )
        return result, Usage(input_tokens=50, output_tokens=10)


def _make_pdf(path: Path) -> None:
    # Headings render larger than body so the font-size gate detects them; two
    # body lines per page keep the body font the most common (the chunk threshold).
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "GOBLIN", fontsize=16)
    p1.insert_text((72, 100), "O goblin resiste a fogo.", fontsize=11)
    p1.insert_text((72, 120), "Pequeno e covarde.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 72), "ORC", fontsize=16)
    p2.insert_text((72, 100), "O orc habita cavernas.", fontsize=11)
    p2.insert_text((72, 120), "Brutal e territorial.", fontsize=11)
    doc.save(path)
    doc.close()


def test_document_extractor_satisfies_the_port() -> None:
    assert isinstance(DocumentExtractor(llm=_FakeLLM()), Extractor)


def test_extract_returns_nodes_and_edges_from_the_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "mm.pdf"
    _make_pdf(pdf)
    nodes, edges = DocumentExtractor(llm=_FakeLLM()).extract(pdf)
    by_id = {n.id: n for n in nodes}
    assert by_id["goblin"].type is NodeType.ENTITY
    assert {e.type for e in edges} == {EdgeType.RESISTS, EdgeType.INHABITS}


def test_extract_calls_the_llm_once_per_creature(tmp_path: Path) -> None:
    pdf = tmp_path / "mm.pdf"
    _make_pdf(pdf)
    llm = _FakeLLM()
    DocumentExtractor(llm=llm).extract(pdf)
    assert len(llm.seen) == 2


def test_extract_with_usage_reports_one_usage_per_chunk(tmp_path: Path) -> None:
    pdf = tmp_path / "mm.pdf"
    _make_pdf(pdf)
    nodes, edges, usages = DocumentExtractor(llm=_FakeLLM()).extract_with_usage(pdf)
    assert len(usages) == 2
    assert nodes and edges


class _FlakyLLM:
    """Raises on the Orc chunk; succeeds on the Goblin chunk."""

    def extract(self, system: str, text: str) -> tuple[ExtractionResult, Usage]:
        if "ORC" in text:
            raise RuntimeError("simulated API failure")
        return (
            ExtractionResult(
                entities=[ExtractedEntity(name="Goblin", kind="creature")],
                relations=[],
            ),
            Usage(input_tokens=50, output_tokens=10),
        )


def test_extract_skips_chunks_that_raise(tmp_path: Path) -> None:
    pdf = tmp_path / "mm.pdf"
    _make_pdf(pdf)
    nodes, _edges, usages = DocumentExtractor(llm=_FlakyLLM()).extract_with_usage(pdf)
    assert len(usages) == 1  # the failing Orc chunk is skipped, not fatal
    assert any(n.id == "goblin" for n in nodes)
