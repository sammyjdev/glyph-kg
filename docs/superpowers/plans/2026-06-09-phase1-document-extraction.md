# Phase 1 — Document Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a knowledge graph of the D&D Monster Manual (PT-BR) from its PDF, behind the existing `Extractor` port, and measure LLM extraction cost on the book (P1.1–P1.4).

**Architecture:** New `glyph/extract/document/` adapter package: `pdf.py` (PyMuPDF text + page metadata) → `chunk.py` (per-creature split by ALL-CAPS heading) → `llm.py` (Claude Haiku 4.5 structured output) → `schema.py` (map to `Node`/`Edge` with dedup). `cost.py` aggregates token usage into a USD cost report. A composition-root script `scripts/extract_book.py` wires extractor + `NetworkXStore` for the paid P1.4 gate run. The adapter never imports `glyph.store` — the Phase 0 architecture invariant test enforces this.

**Tech Stack:** Python 3.11, Pydantic v2, PyMuPDF (`pymupdf`/`fitz`), `anthropic` SDK (Claude Haiku 4.5, `messages.parse` structured outputs), pytest.

**Conventions:**
- TDD per step: write the failing test, watch it fail, minimal implementation, watch it pass, commit.
- Every commit message ends with a second `-m` trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Run `python3 -m pytest` (this environment resolves `pytest` via `python3 -m`). Architecture invariants and coverage gate (90%) must stay green.
- Live API tests are marked `@pytest.mark.live` and deselected by default (`-m "not live"` in addopts). Run them with `python3 -m pytest -m live`.

---

### Task 1: Dependencies, package skeleton, pytest config, ADR-G2

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Create: `glyph/extract/document/__init__.py`
- Create: `docs/decisions/dec-g2-document-extraction-schema.md`
- Create: `tests/extract/document/__init__.py` (empty, keeps the dir present)

- [ ] **Step 1: Add the `document` optional extra and live marker to `pyproject.toml`**

In `[project.optional-dependencies]`, add a `document` group after `dev`:

```toml
document = [
    "pymupdf>=1.24",
    "anthropic>=0.40",
]
```

Replace the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-p no:asyncio --import-mode=importlib -m \"not live\" --cov=glyph --cov-report=term-missing --cov-fail-under=90"
markers = [
    "live: hits the live Anthropic API (requires ANTHROPIC_API_KEY); deselected by default",
]
```

- [ ] **Step 2: Install the document extra**

Run: `python3 -m pip install -e ".[dev,document]"`
Expected: `Successfully installed` (pymupdf, anthropic and deps).

- [ ] **Step 3: Update CI to install the document extra**

In `.github/workflows/ci.yml`, change the Install step:

```yaml
      - name: Install
        run: pip install -e ".[dev,document]"
```

- [ ] **Step 4: Create the document package skeleton**

Create `glyph/extract/document/__init__.py`:

```python
"""DocumentExtractor: build a graph from a PDF via structure-aware chunking + LLM."""
```

Create `tests/extract/document/__init__.py` as an empty file.

- [ ] **Step 5: Write ADR-G2**

Create `docs/decisions/dec-g2-document-extraction-schema.md`:

```markdown
# ADR-G2: Schema de extração documental (Monster Manual)

**Data:** 2026-06-09
**Status:** Aceito

## Contexto

A Fase 1 constrói o primeiro knowledge graph documental do GLYPH a partir do
Monster Manual (PT-BR, 351 páginas, texto extraível, sem TOC). A extração é
probabilística: um LLM lê prosa em português e infere entidades e relações, com
erro. O `EdgeType` do Phase 0 cobre apenas `{RELATES_TO, MENTIONS, REQUIRES,
RESISTS}` para documento — insuficiente para as relações que o Monster Manual
exercita de fato.

## Decisão

**Schema MM-focado.** Entidade = criatura (`NodeType.ENTITY`); tipo de dano,
condição e local/plano = conceito (`NodeType.CONCEPT`). `EdgeType` ganha quatro
membros de domínio documento:

- `IMMUNE_TO` — imunidade a dano/condição
- `VULNERABLE_TO` — vulnerabilidade a dano
- `INHABITS` — habita local/plano
- `SUMMONS` — invoca/conjura outra criatura

somando aos existentes `RESISTS`, `RELATES_TO`, `MENTIONS`, `REQUIRES`.
`NodeType` não muda.

Schema do grafo:

```
ENTITY(criatura) --RESISTS/IMMUNE_TO/VULNERABLE_TO--> CONCEPT(fogo, frio, atordoado, ...)
ENTITY(criatura) --INHABITS--> CONCEPT(Subterrâneo, plano, ...)
ENTITY(criatura) --SUMMONS--> ENTITY(criatura invocada)
```

## Consequências

**Positivas:** schema casa com a estrutura do MM, alto sinal para o benchmark
grafo-vs-vetor. Adicionar um domínio novo (PHB/DMG) é estender o enum e o prompt,
sem tocar o núcleo.

**Trade-offs / a observar:** a extração documental tem erro — é por isso que o
benchmark mede qualidade em vez de assumir. O schema não cobre magia/item/regra
(conteúdo de PHB/DMG); isso é declarado e fica para fase futura.

## Alternativas consideradas

| Alternativa | Por que foi descartada |
|---|---|
| Reusar só os EdgeType existentes | Funde imune/resiste/vulnerável numa aresta só; perde fidelidade |
| Schema amplo (magia/item/regra/local) já | MM não exercita esses tipos; extração ruidosa, mais cara, sem ganho agora |
| Dict de atributos livre na saída do LLM | Structured outputs exigem `additionalProperties:false`; campos fixos opcionais no lugar |
```

- [ ] **Step 6: Verify the suite still passes with the new config**

Run: `python3 -m pytest -q`
Expected: `38 passed`, coverage gate reached (no `glyph/extract/document` modules with code yet).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml glyph/extract/document/__init__.py tests/extract/document/__init__.py docs/decisions/dec-g2-document-extraction-schema.md
git commit -m "Add document-extraction deps, package skeleton, and ADR-G2 (P1)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Extend EdgeType with the document relations

**Files:**
- Modify: `glyph/model/edge.py`
- Test: `tests/model/test_edge.py`

- [ ] **Step 1: Write the failing test**

Replace the body of `test_edgetype_separates_code_and_document_domains` in `tests/model/test_edge.py` with:

```python
def test_edgetype_separates_code_and_document_domains() -> None:
    code = {
        EdgeType.DEFINES,
        EdgeType.IMPORTS,
        EdgeType.CALLS,
        EdgeType.INHERITS,
        EdgeType.REFERENCES,
    }
    document = {
        EdgeType.RELATES_TO,
        EdgeType.MENTIONS,
        EdgeType.REQUIRES,
        EdgeType.RESISTS,
        EdgeType.IMMUNE_TO,
        EdgeType.VULNERABLE_TO,
        EdgeType.INHABITS,
        EdgeType.SUMMONS,
    }
    assert code.isdisjoint(document)
    assert code | document <= set(EdgeType)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/model/test_edge.py::test_edgetype_separates_code_and_document_domains -v`
Expected: FAIL with `AttributeError: IMMUNE_TO` (member doesn't exist yet).

- [ ] **Step 3: Add the four members**

In `glyph/model/edge.py`, in the `EdgeType` enum under the `# document` group, after `RESISTS = "resists"` add:

```python
    IMMUNE_TO = "immune_to"
    VULNERABLE_TO = "vulnerable_to"
    INHABITS = "inhabits"
    SUMMONS = "summons"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/model/ -q`
Expected: PASS (all model tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/model/edge.py tests/model/test_edge.py
git commit -m "Extend EdgeType with document relations IMMUNE_TO/VULNERABLE_TO/INHABITS/SUMMONS (P1, ADR-G2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: PDF ingestion (P1.1)

**Files:**
- Create: `glyph/extract/document/pdf.py`
- Test: `tests/extract/document/test_pdf.py`

- [ ] **Step 1: Write the failing test**

Create `tests/extract/document/test_pdf.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/extract/document/test_pdf.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.extract.document.pdf'`.

- [ ] **Step 3: Implement `pdf.py`**

Create `glyph/extract/document/pdf.py`:

```python
"""P1.1: read a PDF into pages with text, per-line text, and metadata."""

from dataclasses import dataclass
from pathlib import Path

import fitz

from glyph.extract.port import Source


@dataclass(frozen=True)
class Page:
    """One PDF page: full text plus the individual lines (for heading detection)."""

    book: str
    number: int  # 1-based
    text: str
    lines: tuple[str, ...]


def load(source: Source, book: str | None = None) -> list[Page]:
    """Read every page of ``source`` into :class:`Page` objects."""
    path = Path(source)
    book_name = book if book is not None else path.stem
    doc = fitz.open(path)
    try:
        pages: list[Page] = []
        for index in range(doc.page_count):
            page = doc[index]
            data = page.get_text("dict")
            lines: list[str] = []
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line.get("spans", [])).strip()
                    if text:
                        lines.append(text)
            pages.append(
                Page(
                    book=book_name,
                    number=index + 1,
                    text=page.get_text("text"),
                    lines=tuple(lines),
                )
            )
        return pages
    finally:
        doc.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/extract/document/test_pdf.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/extract/document/pdf.py tests/extract/document/test_pdf.py
git commit -m "Add PDF ingestion: pages with text, lines, and metadata (P1.1)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Structure-aware chunking (P1.2)

**Files:**
- Create: `glyph/extract/document/chunk.py`
- Test: `tests/extract/document/test_chunk.py`

- [ ] **Step 1: Write the failing test**

Create `tests/extract/document/test_chunk.py`:

```python
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
        _page(1, ["GOBLIN", "Primeira parte do verbete."]),
        _page(2, ["Continuacao do goblin na pagina dois."]),
        _page(2, []),
    ]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/extract/document/test_chunk.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.extract.document.chunk'`.

- [ ] **Step 3: Implement `chunk.py`**

Create `glyph/extract/document/chunk.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/extract/document/test_chunk.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/extract/document/chunk.py tests/extract/document/test_chunk.py
git commit -m "Add per-creature chunking by heading detection (P1.2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Extraction schema and graph mapping

**Files:**
- Create: `glyph/extract/document/schema.py`
- Test: `tests/extract/document/test_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/extract/document/test_schema.py`:

```python
from glyph.extract.document.schema import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    merge,
)
from glyph.model.edge import EdgeType
from glyph.model.node import NodeType


def _result() -> ExtractionResult:
    return ExtractionResult(
        entities=[
            ExtractedEntity(name="Goblin", kind="creature", creature_type="humanoide"),
            ExtractedEntity(name="fogo", kind="concept"),
        ],
        relations=[ExtractedRelation(subject="Goblin", predicate="RESISTS", object="fogo")],
    )


def test_merge_maps_entities_to_typed_nodes() -> None:
    nodes, _ = merge([_result()])
    by_id = {n.id: n for n in nodes}
    assert by_id["goblin"].type is NodeType.ENTITY
    assert by_id["goblin"].label == "Goblin"
    assert by_id["goblin"].attrs == {"creature_type": "humanoide"}
    assert by_id["fogo"].type is NodeType.CONCEPT


def test_merge_maps_relations_to_typed_edges() -> None:
    _, edges = merge([_result()])
    (edge,) = edges
    assert edge.src == "goblin"
    assert edge.dst == "fogo"
    assert edge.type is EdgeType.RESISTS


def test_merge_creates_concept_node_for_unlisted_relation_object() -> None:
    result = ExtractionResult(
        entities=[ExtractedEntity(name="Orc", kind="creature")],
        relations=[ExtractedRelation(subject="Orc", predicate="INHABITS", object="cavernas")],
    )
    nodes, _ = merge([result])
    by_id = {n.id: n for n in nodes}
    assert by_id["cavernas"].type is NodeType.CONCEPT


def test_merge_dedupes_nodes_and_edges_across_results() -> None:
    nodes, edges = merge([_result(), _result()])
    assert len([n for n in nodes if n.id == "goblin"]) == 1
    assert len(edges) == 1


def test_merge_normalizes_id_by_lowercasing_and_collapsing_space() -> None:
    result = ExtractionResult(
        entities=[ExtractedEntity(name="  Verme   Púrpura ", kind="creature")],
        relations=[],
    )
    (node,) = merge([result])[0]
    assert node.id == "verme púrpura"
    assert node.label == "Verme   Púrpura"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/extract/document/test_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.extract.document.schema'`.

- [ ] **Step 3: Implement `schema.py`**

Create `glyph/extract/document/schema.py`:

```python
"""LLM extraction contract and the pure mapping to the graph model.

Structured outputs forbid open dicts (``additionalProperties`` must be false), so
entity attributes are fixed optional fields rather than a free-form map.
"""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType

Predicate = Literal["RESISTS", "IMMUNE_TO", "VULNERABLE_TO", "INHABITS", "SUMMONS"]
Kind = Literal["creature", "concept"]


class ExtractedEntity(BaseModel):
    name: str
    kind: Kind
    challenge_rating: str | None = None
    creature_type: str | None = None
    alignment: str | None = None


class ExtractedRelation(BaseModel):
    subject: str
    predicate: Predicate
    object: str


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


_NODE_TYPE: dict[str, NodeType] = {
    "creature": NodeType.ENTITY,
    "concept": NodeType.CONCEPT,
}


def _nid(name: str) -> str:
    return " ".join(name.split()).lower()


def _attrs(entity: ExtractedEntity) -> dict[str, str]:
    fields = {
        "challenge_rating": entity.challenge_rating,
        "creature_type": entity.creature_type,
        "alignment": entity.alignment,
    }
    return {k: v for k, v in fields.items() if v is not None}


def merge(results: Iterable[ExtractionResult]) -> tuple[list[Node], list[Edge]]:
    """Map and deduplicate extraction results into nodes and edges."""
    nodes: dict[str, Node] = {}
    edges: dict[tuple[str, str, EdgeType], Edge] = {}
    for result in results:
        for entity in result.entities:
            nid = _nid(entity.name)
            if nid not in nodes:
                nodes[nid] = Node(
                    id=nid,
                    type=_NODE_TYPE[entity.kind],
                    label=entity.name.strip(),
                    attrs=_attrs(entity),
                )
        for relation in result.relations:
            src, dst = _nid(relation.subject), _nid(relation.object)
            for endpoint, raw in ((src, relation.subject), (dst, relation.object)):
                if endpoint not in nodes:
                    nodes[endpoint] = Node(
                        id=endpoint, type=NodeType.CONCEPT, label=raw.strip(), attrs={}
                    )
            edge_type = EdgeType[relation.predicate]
            key = (src, dst, edge_type)
            if key not in edges:
                edges[key] = Edge(src=src, dst=dst, type=edge_type)
    return list(nodes.values()), list(edges.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/extract/document/test_schema.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/extract/document/schema.py tests/extract/document/test_schema.py
git commit -m "Add extraction schema and pure graph mapping with dedup" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Extraction prompt

**Files:**
- Create: `glyph/extract/document/prompt.py`
- Test: `tests/extract/document/test_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/extract/document/test_prompt.py`:

```python
from glyph.extract.document.prompt import system_prompt


def test_system_prompt_names_every_predicate() -> None:
    text = system_prompt()
    for predicate in ("RESISTS", "IMMUNE_TO", "VULNERABLE_TO", "INHABITS", "SUMMONS"):
        assert predicate in text


def test_system_prompt_forbids_inventing_relations() -> None:
    text = system_prompt().lower()
    assert "não invente" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/extract/document/test_prompt.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.extract.document.prompt'`.

- [ ] **Step 3: Implement `prompt.py`**

Create `glyph/extract/document/prompt.py`:

```python
"""System prompt for entity/relation extraction from Monster Manual entries (PT-BR)."""

_SYSTEM = """Você extrai um knowledge graph do Manual dos Monstros de Dungeons & Dragons.
Recebe o texto de um verbete de criatura em português e devolve entidades e relações.

Entidades (entities):
- a criatura descrita, com kind="creature" (preencha challenge_rating, creature_type e
  alignment quando o texto os trouxer);
- os conceitos citados, com kind="concept": tipos de dano (fogo, frio, veneno...),
  condições (atordoado, enfeitiçado...) e locais/planos (Subterrâneo, Abismo...).

Relações (relations), ligando a criatura aos conceitos/criaturas por um predicate:
- RESISTS: resistência a um tipo de dano;
- IMMUNE_TO: imunidade a dano ou condição;
- VULNERABLE_TO: vulnerabilidade a um tipo de dano;
- INHABITS: a criatura habita um local ou plano;
- SUMMONS: a criatura invoca ou conjura outra criatura.

Use os nomes exatamente como aparecem no texto. Não invente relações: se o texto não
afirma uma relação, não a inclua."""


def system_prompt() -> str:
    return _SYSTEM
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/extract/document/test_prompt.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/extract/document/prompt.py tests/extract/document/test_prompt.py
git commit -m "Add PT-BR extraction system prompt" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: LLM adapter (Anthropic Haiku) + Usage

**Files:**
- Create: `glyph/extract/document/llm.py`
- Test: `tests/extract/document/test_llm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/extract/document/test_llm.py`:

```python
import os

import pytest

from glyph.extract.document.llm import AnthropicExtractor, Usage
from glyph.extract.document.schema import ExtractionResult


class _FakeMessages:
    def __init__(self, parsed: ExtractionResult) -> None:
        self._parsed = parsed
        self.calls: list[dict] = []

    def parse(self, **kwargs: object):
        self.calls.append(kwargs)

        class _Usage:
            input_tokens = 100
            output_tokens = 20

        class _Resp:
            parsed_output = self._parsed
            usage = _Usage()

        return _Resp()


class _FakeClient:
    def __init__(self, parsed: ExtractionResult) -> None:
        self.messages = _FakeMessages(parsed)


def test_anthropic_extractor_returns_parsed_output_and_usage() -> None:
    parsed = ExtractionResult(entities=[], relations=[])
    client = _FakeClient(parsed)
    extractor = AnthropicExtractor(client=client)
    result, usage = extractor.extract("system text", "verbete")
    assert result is parsed
    assert usage == Usage(input_tokens=100, output_tokens=20)


def test_anthropic_extractor_uses_haiku_by_default() -> None:
    client = _FakeClient(ExtractionResult(entities=[], relations=[]))
    AnthropicExtractor(client=client).extract("s", "t")
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


@pytest.mark.live
def test_anthropic_extractor_live_smoke() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from glyph.extract.document.prompt import system_prompt

    extractor = AnthropicExtractor()
    text = "GOBLIN\nO goblin é um humanoide pequeno que resiste a fogo e habita cavernas."
    result, usage = extractor.extract(system_prompt(), text)
    assert usage.input_tokens > 0
    assert any(r.predicate == "RESISTS" for r in result.relations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/extract/document/test_llm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.extract.document.llm'`. (The live test is deselected by `-m "not live"`.)

- [ ] **Step 3: Implement `llm.py`**

Create `glyph/extract/document/llm.py`:

```python
"""Thin Anthropic adapter: Claude Haiku 4.5 structured-output extraction + token usage."""

from dataclasses import dataclass
from typing import Protocol

from glyph.extract.document.schema import ExtractionResult


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


class LLMExtractor(Protocol):
    def extract(self, system: str, text: str) -> tuple[ExtractionResult, Usage]: ...


class AnthropicExtractor:
    """Calls ``messages.parse`` with the extraction schema and reports token usage."""

    def __init__(self, model: str = "claude-haiku-4-5", client: object | None = None) -> None:
        if client is None:  # pragma: no cover - real wiring, exercised by the live smoke test
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def extract(self, system: str, text: str) -> tuple[ExtractionResult, Usage]:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": text}],
            output_format=ExtractionResult,
        )
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.parsed_output, usage
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/extract/document/test_llm.py -q`
Expected: PASS (2 selected; 1 live deselected).

- [ ] **Step 5: Commit**

```bash
git add glyph/extract/document/llm.py tests/extract/document/test_llm.py
git commit -m "Add Anthropic Haiku extraction adapter with usage capture (P1.3)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: DocumentExtractor (orchestration, implements the Extractor port)

**Files:**
- Create: `glyph/extract/document/extractor.py`
- Modify: `glyph/extract/document/__init__.py`
- Test: `tests/extract/document/test_extractor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/extract/document/test_extractor.py`:

```python
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
                relations=[ExtractedRelation(subject="Orc", predicate="INHABITS", object="cavernas")],
            )
        return result, Usage(input_tokens=50, output_tokens=10)


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "GOBLIN\nO goblin resiste a fogo.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 72), "ORC\nO orc habita cavernas.", fontsize=11)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/extract/document/test_extractor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.extract.document.extractor'`.

- [ ] **Step 3: Implement `extractor.py`**

Create `glyph/extract/document/extractor.py`:

```python
"""DocumentExtractor: orchestrate pdf -> chunk -> LLM -> graph, behind the Extractor port."""

from collections.abc import Sequence

from glyph.extract.document import chunk as chunking
from glyph.extract.document import pdf, prompt
from glyph.extract.document.llm import AnthropicExtractor, LLMExtractor, Usage
from glyph.extract.document.schema import ExtractionResult, merge
from glyph.extract.port import Source
from glyph.model.edge import Edge
from glyph.model.node import Node


class DocumentExtractor:
    """Probabilistic extractor: reads a PDF and infers entities/relations with an LLM."""

    def __init__(self, llm: LLMExtractor | None = None) -> None:
        self._llm = llm if llm is not None else AnthropicExtractor()

    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        nodes, edges, _ = self.extract_with_usage(source)
        return nodes, edges

    def extract_with_usage(
        self, source: Source
    ) -> tuple[list[Node], list[Edge], list[Usage]]:
        pages = pdf.load(source)
        chunks = chunking.by_creature(pages)
        system = prompt.system_prompt()
        results: list[ExtractionResult] = []
        usages: list[Usage] = []
        for piece in chunks:
            result, usage = self._llm.extract(system, piece.text)
            results.append(result)
            usages.append(usage)
        nodes, edges = merge(results)
        return nodes, edges, usages
```

- [ ] **Step 4: Export DocumentExtractor from the subpackage**

Replace `glyph/extract/document/__init__.py` with:

```python
"""DocumentExtractor: build a graph from a PDF via structure-aware chunking + LLM."""

from glyph.extract.document.extractor import DocumentExtractor

__all__ = ["DocumentExtractor"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/extract/document/test_extractor.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add glyph/extract/document/extractor.py glyph/extract/document/__init__.py tests/extract/document/test_extractor.py
git commit -m "Add DocumentExtractor orchestration behind the Extractor port (P1.3)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Cost report (P1.4 measurement)

**Files:**
- Create: `glyph/extract/document/cost.py`
- Test: `tests/extract/document/test_cost.py`

- [ ] **Step 1: Write the failing test**

Create `tests/extract/document/test_cost.py`:

```python
from glyph.extract.document.cost import CostReport, summarize
from glyph.extract.document.llm import Usage


def test_summarize_sums_tokens_and_counts_chunks() -> None:
    report = summarize([Usage(1000, 200), Usage(2000, 300)])
    assert report.chunks == 2
    assert report.input_tokens == 3000
    assert report.output_tokens == 500


def test_summarize_prices_at_haiku_rates() -> None:
    # 1,000,000 input @ $1/M + 1,000,000 output @ $5/M = $6.00
    report = summarize([Usage(1_000_000, 1_000_000)])
    assert report.cost_usd == 6.0


def test_summarize_handles_empty() -> None:
    report = summarize([])
    assert report == CostReport(chunks=0, input_tokens=0, output_tokens=0, cost_usd=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/extract/document/test_cost.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.extract.document.cost'`.

- [ ] **Step 3: Implement `cost.py`**

Create `glyph/extract/document/cost.py`:

```python
"""P1.4: aggregate token usage into a USD cost report at Claude Haiku 4.5 rates."""

from collections.abc import Iterable
from dataclasses import dataclass

from glyph.extract.document.llm import Usage

# Claude Haiku 4.5 pricing, USD per 1M tokens.
INPUT_PER_MILLION = 1.0
OUTPUT_PER_MILLION = 5.0


@dataclass(frozen=True)
class CostReport:
    chunks: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def summarize(usages: Iterable[Usage]) -> CostReport:
    items = list(usages)
    input_tokens = sum(u.input_tokens for u in items)
    output_tokens = sum(u.output_tokens for u in items)
    cost = (
        input_tokens / 1_000_000 * INPUT_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_PER_MILLION
    )
    return CostReport(
        chunks=len(items),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 4),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/extract/document/test_cost.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite and all gates**

Run: `python3 -m pytest -q`
Expected: PASS, total coverage ≥ 90%.

Run: `python3 -m ruff check glyph tests && python3 -m ruff format --check glyph tests && python3 -m mypy glyph`
Expected: all clean.

If `mypy` flags the fake-client `object` type in `llm.py` usage access during tests, that is test-only code; production `llm.py` accesses `response.usage` on the SDK return — leave a `# type: ignore[attr-defined]` only if mypy (which checks `glyph/`, not `tests/`) actually reports it. Re-run `python3 -m mypy glyph` to confirm clean.

- [ ] **Step 6: Commit**

```bash
git add glyph/extract/document/cost.py tests/extract/document/test_cost.py
git commit -m "Add token-usage cost report at Haiku rates (P1.4)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Cost-gate runner + paid P1.4 run

The runner is the composition root — it is the only place that wires the extractor to `NetworkXStore`. It lives in `scripts/` (outside the `glyph` package) so it does not violate the architecture invariant that `glyph.extract` never imports `glyph.store`, and so it is excluded from coverage.

**Files:**
- Create: `scripts/extract_book.py`

- [ ] **Step 1: Implement the runner**

Create `scripts/extract_book.py`:

```python
"""P1.4 cost gate: extract a book, report cost/latency/volume, persist the graph.

Usage:
    ANTHROPIC_API_KEY=... python3 scripts/extract_book.py "<book.pdf>" out/graph.json

Runs the live Anthropic API and costs money. Do not run without explicit approval.
"""

import sys
import time
from pathlib import Path

from glyph.extract.document.cost import summarize
from glyph.extract.document.extractor import DocumentExtractor
from glyph.model.node import NodeType
from glyph.store.networkx_store import NetworkXStore


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    book_path, out_path = argv[1], argv[2]

    started = time.monotonic()
    nodes, edges, usages = DocumentExtractor().extract_with_usage(book_path)
    elapsed = time.monotonic() - started

    report = summarize(usages)
    creatures = [n for n in nodes if n.type is NodeType.ENTITY]

    print(f"book:     {book_path}")
    print(f"chunks:   {report.chunks}")
    print(f"nodes:    {len(nodes)} ({len(creatures)} creatures)")
    print(f"edges:    {len(edges)}")
    print(f"tokens:   in={report.input_tokens} out={report.output_tokens}")
    print(f"cost:     ${report.cost_usd:.4f}")
    print(f"latency:  {elapsed:.1f}s ({elapsed / max(report.chunks, 1):.2f}s/chunk)")

    print("\nsample creatures (up to 10):")
    for node in creatures[:10]:
        out = [e for e in edges if e.src == node.id]
        rels = ", ".join(f"{e.type.value} {e.dst}" for e in out[:5])
        print(f"  - {node.label}: {rels or '(no relations)'}")

    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges(edges)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    store.save(Path(out_path))
    print(f"\npersisted graph -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Commit the runner**

```bash
git add scripts/extract_book.py
git commit -m "Add cost-gate runner script for P1.4" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Run the live smoke test (one paid call, requires approval + key)**

Confirm with the user, then:

Run: `ANTHROPIC_API_KEY=... python3 -m pytest -m live tests/extract/document/test_llm.py -q`
Expected: PASS (1 test) — proves the real `messages.parse` path returns parsed output and usage.

- [ ] **Step 4: Run the P1.4 cost gate on the Monster Manual (paid, requires approval)**

Confirm with the user that the estimated ~$1–3 spend is approved, then:

Run:
```bash
ANTHROPIC_API_KEY=... python3 scripts/extract_book.py "$HOME/Downloads/3-Monster Manual.pdf" out/monster-manual.json
```
Expected: a printed report (chunks, nodes/creatures, edges, tokens, **cost in USD**, latency, ~10 sampled creatures with their relations) and `persisted graph -> out/monster-manual.json`.

- [ ] **Step 5: Review the gate output (manual quality check)**

Read the sampled creatures and their relations. Confirm the relations are plausible against the book (e.g. a creature that "resiste a fogo" shows `resists fogo`). Record the cost, latency, node/edge counts, and a one-paragraph quality note. **Do not scale to PHB/DMG — that is P1.5, out of scope for this session.**

- [ ] **Step 6: Commit the gate results note**

Create `docs/decisions/phase1-cost-gate-results.md` with the recorded numbers and quality note, then:

```bash
git add docs/decisions/phase1-cost-gate-results.md out/monster-manual.json
git commit -m "Record P1.4 cost-gate results on the Monster Manual" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- P1.1 PDF ingestion → Task 3. ✓
- P1.2 chunking → Task 4. ✓
- P1.3 DocumentExtractor (schema, prompt, LLM, orchestration) → Tasks 5–8. ✓
- P1.4 cost gate (measurement + paid run) → Tasks 9–10. ✓
- ADR-G2 + EdgeType extension → Tasks 1–2. ✓
- `[document]` extra, CI, pytest live marker → Task 1. ✓
- TDD with fake LLM + `@pytest.mark.live` smoke → Tasks 7–8, 10. ✓
- Architecture invariant preserved (runner in `scripts/`, not in `glyph.extract`) → Task 10. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `Usage(input_tokens, output_tokens)`, `ExtractionResult{entities, relations}`, `ExtractedEntity{name, kind, challenge_rating, creature_type, alignment}`, `ExtractedRelation{subject, predicate, object}`, `merge(...) -> (nodes, edges)`, `DocumentExtractor.extract` / `extract_with_usage`, `summarize(...) -> CostReport`, `Page{book, number, text, lines}`, `Chunk{label, text, book, pages}` — names used identically across tasks. `EdgeType[predicate]` keys by enum member name (uppercase), matching the `Predicate` literals. ✓

**Note on coverage:** `AnthropicExtractor.extract` is exercised offline by `_FakeClient`; only the `client is None` real-wiring branch is `# pragma: no cover` and is covered by the live smoke test. The `scripts/` runner is outside `--cov=glyph` source, so it does not affect the 90% gate.
