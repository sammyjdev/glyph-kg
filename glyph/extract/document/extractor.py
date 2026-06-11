"""DocumentExtractor: orchestrate pdf -> chunk -> LLM -> graph, behind the Extractor port."""

from collections.abc import Callable, Sequence

from glyph.extract.document import chunk as chunking
from glyph.extract.document import pdf, prompt
from glyph.extract.document.llm import AnthropicExtractor, LLMExtractor, Usage
from glyph.extract.document.schema import ExtractionResult, merge
from glyph.extract.port import Source
from glyph.model.edge import Edge
from glyph.model.node import Node


class DocumentExtractor:
    """Probabilistic extractor: reads a PDF and infers entities/relations with an LLM."""

    def __init__(
        self,
        llm: LLMExtractor | None = None,
        keep: Callable[[chunking.Chunk], bool] | None = None,
    ) -> None:
        self._llm = llm if llm is not None else AnthropicExtractor()
        self._keep = keep

    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        nodes, edges, _ = self.extract_with_usage(source)
        return nodes, edges

    def extract_with_usage(self, source: Source) -> tuple[list[Node], list[Edge], list[Usage]]:
        pages = pdf.load(source)
        chunks = chunking.by_creature(pages)
        if self._keep is not None:
            chunks = [piece for piece in chunks if self._keep(piece)]
        system = prompt.system_prompt()
        results: list[ExtractionResult] = []
        usages: list[Usage] = []
        for piece in chunks:
            try:
                result, usage = self._llm.extract(system, piece.text)
            except Exception:  # noqa: BLE001 - one failed chunk must not abort a paid run
                continue
            results.append(result)
            usages.append(usage)
        nodes, edges = merge(results)
        return nodes, edges, usages
