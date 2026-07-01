# GLYPH: Retrieval + Insertion Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar três capacidades ao GLYPH: extração determinística de relações de documentos Markdown (MarkdownRelationExtractor), retrieval por caminho mais curto entre duas entidades (PathRetriever) e retrieval multi-âncora com extração de entidades da query (MultiAnchorRetriever).

**Architecture:** MarkdownRelationExtractor satisfaz o port `Extractor` existente e produz arestas REQUIRES/RELATES_TO/REFERENCES a partir de frontmatter YAML e links Markdown, sem custo de LLM. PathRetriever e MultiAnchorRetriever satisfazem o port `Retriever` existente e delegam ao `GraphStore` já disponível (`shortest_path` e `subgraph`). Nenhum dos três requer nova dependência.

**Tech Stack:** Python 3.11+, NetworkX (via `GraphStore`), `InMemoryVectorIndex` (já disponível em `glyph/embed/memory_index.py`), pytest, `glyph.model.{Node, Edge, NodeType, EdgeType}`.

## Global Constraints

- Zero novas dependências externas — stdlib e pacotes já instalados apenas.
- Todos os novos retrievers satisfazem `glyph.retrieval.port.Retriever` (Protocol): `retrieve(query: str, token_budget: int) -> ContextPack`.
- `MarkdownRelationExtractor` satisfaz `glyph.extract.port.Extractor` (Protocol): `extract(source: Source) -> tuple[Sequence[Node], Sequence[Edge]]`.
- `Source = Path | str` — extractor requer `Path`; passar `str` levanta `TypeError`.
- Overlay de community (`NodeType.COMMUNITY` / `EdgeType.CONTAINS`) deve ser excluído de todas as traversals (padrão estabelecido em `graph.py`, dec-g7).
- TDD obrigatório: test RED antes de qualquer implementação.
- `pytest tests/ -q` deve passar a 100% ao fim de cada task.

---

## Test Coverage Matrix

| Story ID | Requirement | Test IDs | Status |
|---|---|---|---|
| S-01 | MarkdownRelationExtractor extrai relações de frontmatter (inline list, block list, scalar) | `test_s01_*` | ✗ unmet |
| S-02 | MarkdownRelationExtractor extrai referências de links Markdown (exclui http, âncoras) | `test_s02_*` | ✗ unmet |
| S-03 | MarkdownRelationExtractor produz Node+Edge válidos; nó do documento sempre presente se há relações | `test_s03_*` | ✗ unmet |
| S-04 | MarkdownRelationExtractor retorna vazio quando não há frontmatter nem links | `test_s04_empty` | ✗ unmet |
| S-05 | MarkdownRelationExtractor satisfaz o port `Extractor` (isinstance check) | `test_s05_port` | ✗ unmet |
| S-06 | PathRetriever retorna ContextPack com segmentos do caminho quando há path entre as duas âncoras | `test_s06_*` | ✗ unmet |
| S-07 | PathRetriever retorna as duas âncoras como fallback quando não há caminho | `test_s07_no_path_fallback` | ✗ unmet |
| S-08 | PathRetriever satisfaz o port `Retriever` (isinstance check) | `test_s08_port` | ✗ unmet |
| S-09 | PathRetriever exclui overlay de community da traversal | `test_s09_overlay_excluded` | ✗ unmet |
| S-10 | MultiAnchorRetriever extrai entidades nomeadas da query (CamelCase, acrônimos, padrões dec-/ADR-) | `test_s10_*` | ✗ unmet |
| S-11 | MultiAnchorRetriever expande subgraph de todas as âncoras simultaneamente | `test_s11_multi_seed` | ✗ unmet |
| S-12 | MultiAnchorRetriever faz fallback para top-N por embedding quando query não tem entidades nomeadas | `test_s12_fallback` | ✗ unmet |
| S-13 | MultiAnchorRetriever satisfaz o port `Retriever` | `test_s13_port` | ✗ unmet |

---

## File Structure

```
glyph/
  extract/
    document/
      md_relation_extractor.py   ← CRIAR  (S-01..S-05)
  retrieval/
    path.py                      ← CRIAR  (S-06..S-09)
    multi_anchor.py              ← CRIAR  (S-10..S-13)
    __init__.py                  ← MODIFICAR: exportar novos retrievers
tests/
  extract/
    document/
      test_md_relation_extractor.py  ← CRIAR
  retrieval/
    test_path_retriever.py           ← CRIAR
    test_multi_anchor_retriever.py   ← CRIAR
```

---

## Task 1: MarkdownRelationExtractor

**Files:**
- Create: `glyph/extract/document/md_relation_extractor.py`
- Create: `tests/extract/document/test_md_relation_extractor.py`

**Interfaces:**
- Produces: `MarkdownRelationExtractor` class com `extract(source: Path) -> tuple[Sequence[Node], Sequence[Edge]]`
- Mapping de campos:
  - `relates_to`, `extends`, `supersedes`, `replaces` → `EdgeType.RELATES_TO`
  - `requires`, `implements` → `EdgeType.REQUIRES`
  - Links Markdown locais → `EdgeType.REFERENCES`
  - `supersedes`/`replaces` adicionam `attrs={"verb": field}` na aresta

- [ ] **Step 1: Criar o arquivo de teste com os casos RED**

```python
# tests/extract/document/test_md_relation_extractor.py
"""S-01..S-05: MarkdownRelationExtractor deterministic relation extraction."""
import pytest
from pathlib import Path
from glyph.extract.document.md_relation_extractor import MarkdownRelationExtractor
from glyph.extract.port import Extractor
from glyph.model.edge import EdgeType
from glyph.model.node import NodeType


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# --- S-01: frontmatter relations ---

def test_s01_inline_list(tmp_path):
    p = _write(tmp_path, "dec-002.md",
        "---\nrelates_to: [dec-001, ADR-G6]\n---\n# Body\nsome text\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    edge_dsts = {e.dst for e in edges}
    assert "dec-001" in edge_dsts
    assert "ADR-G6" in edge_dsts
    assert all(e.type == EdgeType.RELATES_TO for e in edges if e.dst in ("dec-001", "ADR-G6"))


def test_s01_block_list(tmp_path):
    p = _write(tmp_path, "dec-003.md",
        "---\nsupersedes:\n  - dec-001\n  - dec-002\n---\n# Body\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    assert {e.dst for e in edges} == {"dec-001", "dec-002"}
    assert all(e.attrs.get("verb") == "supersedes" for e in edges)


def test_s01_scalar_requires(tmp_path):
    p = _write(tmp_path, "adr-007.md",
        "---\nrequires: PostgreSQL\n---\n# Body\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert any(e.dst == "PostgreSQL" and e.type == EdgeType.REQUIRES for e in edges)


def test_s01_unknown_fields_ignored(tmp_path):
    p = _write(tmp_path, "doc.md",
        "---\nid: doc\nstatus: accepted\ndate: 2026-06-30\n---\n# Body\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    assert list(nodes) == [] and list(edges) == []


# --- S-02: link references ---

def test_s02_local_links_extracted(tmp_path):
    p = _write(tmp_path, "doc.md",
        "# Body\nSee [ADR-G7](../decisions/dec-g7.md) and [dec-121](./dec-121.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    dsts = {e.dst for e in edges}
    assert "dec-g7" in dsts and "dec-121" in dsts
    assert all(e.type == EdgeType.REFERENCES for e in edges)


def test_s02_http_links_excluded(tmp_path):
    p = _write(tmp_path, "doc.md",
        "# Body\n[GitHub](https://github.com/foo) and [local](./file.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert {e.dst for e in edges} == {"file"}


def test_s02_anchor_links_excluded(tmp_path):
    p = _write(tmp_path, "doc.md",
        "# Body\n[section](#heading) and [other](./file.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert {e.dst for e in edges} == {"file"}


def test_s02_self_reference_excluded(tmp_path):
    p = _write(tmp_path, "dec-001.md",
        "# Body\n[self](./dec-001.md) and [other](./dec-002.md)\n")
    _, edges = MarkdownRelationExtractor().extract(p)
    assert all(e.dst != "dec-001" for e in edges)


# --- S-03: node shape ---

def test_s03_document_node_present(tmp_path):
    p = _write(tmp_path, "dec-005.md",
        "---\nrelates_to: [dec-004]\n---\n# Body\n")
    nodes, _ = MarkdownRelationExtractor().extract(p)
    ids = {n.id for n in nodes}
    assert "dec-005" in ids
    doc_node = next(n for n in nodes if n.id == "dec-005")
    assert doc_node.type == NodeType.ENTITY


def test_s03_target_nodes_created(tmp_path):
    p = _write(tmp_path, "dec-005.md",
        "---\nrelates_to: [dec-004]\nrequires: PostgreSQL\n---\n# Body\n")
    nodes, _ = MarkdownRelationExtractor().extract(p)
    ids = {n.id for n in nodes}
    assert "dec-004" in ids and "PostgreSQL" in ids


def test_s03_no_duplicate_nodes(tmp_path):
    p = _write(tmp_path, "dec-005.md",
        "---\nrelates_to: [dec-004]\n---\n# See [dec-004](./dec-004.md)\n")
    nodes, _ = MarkdownRelationExtractor().extract(p)
    ids = [n.id for n in nodes]
    assert len(ids) == len(set(ids))


# --- S-04: empty document ---

def test_s04_empty_returns_nothing(tmp_path):
    p = _write(tmp_path, "plain.md", "# Just a heading\nNo relations here.\n")
    nodes, edges = MarkdownRelationExtractor().extract(p)
    assert list(nodes) == [] and list(edges) == []


# --- S-05: port conformance ---

def test_s05_port(tmp_path):
    assert isinstance(MarkdownRelationExtractor(), Extractor)
```

- [ ] **Step 2: Rodar os testes para confirmar RED**

```bash
cd glyph-kg && pytest tests/extract/document/test_md_relation_extractor.py -v --no-header 2>&1 | tail -20
```
Esperado: `ImportError: cannot import name 'MarkdownRelationExtractor'`

- [ ] **Step 3: Implementar `md_relation_extractor.py`**

```python
# glyph/extract/document/md_relation_extractor.py
"""Deterministic relation extractor for Markdown documents.

Reads YAML frontmatter fields and local Markdown links to produce
REQUIRES/RELATES_TO/REFERENCES edges without any LLM call.
Satisfies the Extractor port (Source = Path | str; requires Path).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from glyph.extract.port import Source
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

_RELATION_FIELDS = frozenset(
    {"relates_to", "requires", "supersedes", "implements", "extends", "replaces"}
)

_FIELD_TO_EDGE: dict[str, EdgeType] = {
    "relates_to": EdgeType.RELATES_TO,
    "extends": EdgeType.RELATES_TO,
    "supersedes": EdgeType.RELATES_TO,
    "replaces": EdgeType.RELATES_TO,
    "requires": EdgeType.REQUIRES,
    "implements": EdgeType.REQUIRES,
}


def _frontmatter_body(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    return m.group(1) if m else ""


def _strip_frontmatter(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    return text[m.end():] if m else text


def _parse_relations(fm: str) -> list[tuple[str, str]]:
    """Return (field, target) pairs from known relation fields in YAML frontmatter."""
    relations: list[tuple[str, str]] = []
    current_field: str | None = None
    for line in fm.splitlines():
        if current_field and line.strip().startswith("- "):
            target = line.strip()[2:].strip().strip("'\"")
            if target:
                relations.append((current_field, target))
            continue
        current_field = None
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        if key not in _RELATION_FIELDS:
            continue
        val = val.strip()
        if not val:
            current_field = key
        else:
            for item in val.strip("[]").split(","):
                item = item.strip().strip("'\"")
                if item:
                    relations.append((key, item))
    return relations


def _parse_references(body: str) -> list[str]:
    """Extract local file stems from markdown links, preserving order."""
    seen: set[str] = set()
    refs: list[str] = []
    for href in _LINK_RE.findall(body):
        if href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        stem = Path(href.split("#")[0].split("?")[0]).stem
        if stem and stem not in seen:
            seen.add(stem)
            refs.append(stem)
    return refs


class MarkdownRelationExtractor:
    """Deterministic extractor: frontmatter fields + local links → graph edges.

    Requires a Path as source (reads file + derives document ID from stem).
    Zero LLM cost. Complements DocumentExtractor (LLM) which handles entity
    extraction from prose and implicit relations.
    """

    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        if not isinstance(source, Path):
            source = Path(source)
        text = source.read_text(encoding="utf-8")
        doc_id = source.stem

        fm = _frontmatter_body(text)
        body = _strip_frontmatter(text)

        relations = _parse_relations(fm)
        references = _parse_references(body)

        if not relations and not references:
            return [], []

        nodes: list[Node] = [Node(id=doc_id, type=NodeType.ENTITY, label=doc_id)]
        seen_ids: set[str] = {doc_id}
        edges: list[Edge] = []

        for field, target in relations:
            if target not in seen_ids:
                nodes.append(Node(id=target, type=NodeType.ENTITY, label=target))
                seen_ids.add(target)
            edge_type = _FIELD_TO_EDGE.get(field, EdgeType.RELATES_TO)
            attrs = {"verb": field} if field in ("supersedes", "replaces") else {}
            edges.append(Edge(src=doc_id, dst=target, type=edge_type, attrs=attrs))

        for ref in references:
            if ref == doc_id:
                continue
            if ref not in seen_ids:
                nodes.append(Node(id=ref, type=NodeType.ENTITY, label=ref))
                seen_ids.add(ref)
            edges.append(Edge(src=doc_id, dst=ref, type=EdgeType.REFERENCES))

        return nodes, edges
```

- [ ] **Step 4: Rodar testes para confirmar GREEN**

```bash
pytest tests/extract/document/test_md_relation_extractor.py -v --no-header 2>&1 | tail -15
```
Esperado: todos os `test_s01_*` .. `test_s05_*` passando.

- [ ] **Step 5: Rodar suite completa para confirmar sem regressão**

```bash
pytest tests/ -q --no-header 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add glyph/extract/document/md_relation_extractor.py \
        tests/extract/document/test_md_relation_extractor.py
git commit -m "feat(extract): MarkdownRelationExtractor — deterministic frontmatter+link edges"
```

---

## Task 2: PathRetriever

**Files:**
- Create: `glyph/retrieval/path.py`
- Create: `tests/retrieval/test_path_retriever.py`

**Interfaces:**
- Consumes: `GraphStore.shortest_path(src, dst, exclude_node_types, exclude_edge_types) -> Path | None`
- Consumes: `GraphStore.subgraph(seed, hops=0, ...) -> Subgraph` — para obter edge types ao longo do path
- Consumes: `InMemoryVectorIndex` de `glyph.embed.memory_index`
- Produces: `PathRetriever(store, embedder, nodes)` com `retrieve(query, token_budget) -> ContextPack`
- `ContextPack.mode` = `"graph"` (reutiliza o mode existente — PathRetriever é um arm de grafo)

**Comportamento:**
1. Embed query → top-2 âncoras por similaridade de label
2. `store.shortest_path(anchor_0, anchor_1, exclude_overlay)`
3. Path encontrado → `store.subgraph(path.nodes, hops=0, exclude_overlay)` para obter edge types entre nós consecutivos → render como `"A —[relates_to]→ B —[requires]→ C"`
4. Sem path → retorna as duas âncoras como segmentos isolados (fallback)

- [ ] **Step 1: Criar o arquivo de teste com os casos RED**

```python
# tests/retrieval/test_path_retriever.py
"""S-06..S-09: PathRetriever — shortest-path arm."""
import pytest
from unittest.mock import MagicMock
from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.model.contract import ContextPack
from glyph.model.edge import Edge, EdgeType
from glyph.model.graph import Path, Subgraph
from glyph.model.node import Node, NodeType
from glyph.retrieval.path import PathRetriever
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


def _make_store() -> NetworkXStore:
    store = NetworkXStore()
    nodes = [
        Node(id="A", type=NodeType.ENTITY, label="Alpha"),
        Node(id="B", type=NodeType.ENTITY, label="Beta"),
        Node(id="C", type=NodeType.ENTITY, label="Gamma"),
    ]
    edges = [
        Edge(src="A", dst="B", type=EdgeType.REQUIRES),
        Edge(src="B", dst="C", type=EdgeType.RELATES_TO),
    ]
    store.upsert_nodes(nodes)
    store.upsert_edges(edges)
    return store


def _make_embedder(mapping: dict[str, list[float]]):
    """Stub embedder: returns a fixed vector per text."""
    emb = MagicMock()
    emb.embed.side_effect = lambda texts: [mapping[t] for t in texts]
    return emb


# --- S-06: happy path ---

def test_s06_path_found_returns_intermediate_nodes():
    # Query embedding closest to "Alpha" and "Gamma"; path A→B→C exists.
    store = _make_store()
    mapping = {
        "Alpha": [1.0, 0.0],
        "Beta":  [0.0, 1.0],
        "Gamma": [0.9, 0.1],
        "find path from Alpha to Gamma": [1.0, 0.0],
    }
    embedder = _make_embedder(mapping)
    retriever = PathRetriever(store, embedder, store._g.nodes)

    pack = retriever.retrieve("find path from Alpha to Gamma", token_budget=2000)

    assert isinstance(pack, ContextPack)
    sources = {s.source for s in pack.segments}
    assert "A" in sources  # source anchor
    assert "C" in sources  # dest anchor
    assert "B" in sources  # intermediate


def test_s06_path_segments_include_edge_type():
    store = _make_store()
    mapping = {
        "Alpha": [1.0, 0.0], "Beta": [0.0, 1.0], "Gamma": [0.9, 0.1],
        "q": [1.0, 0.0],
    }
    embedder = _make_embedder(mapping)
    # Reconstruct nodes list from store
    nodes = [Node(id=nid, type=NodeType.ENTITY, label=store._g.nodes[nid]["label"])
             for nid in store._g.nodes]
    retriever = PathRetriever(store, embedder, nodes)
    pack = retriever.retrieve("q", token_budget=2000)
    joined = " ".join(s.text for s in pack.segments)
    assert "requires" in joined or "relates_to" in joined


# --- S-07: no path fallback ---

def test_s07_no_path_fallback_returns_anchors():
    store = NetworkXStore()
    store.upsert_nodes([
        Node(id="X", type=NodeType.ENTITY, label="Xray"),
        Node(id="Y", type=NodeType.ENTITY, label="Yankee"),
    ])
    # No edges — no path between X and Y
    mapping = {"Xray": [1.0, 0.0], "Yankee": [0.0, 1.0], "q": [0.6, 0.4]}
    embedder = _make_embedder(mapping)
    nodes = [Node(id="X", type=NodeType.ENTITY, label="Xray"),
             Node(id="Y", type=NodeType.ENTITY, label="Yankee")]
    retriever = PathRetriever(store, embedder, nodes)
    pack = retriever.retrieve("q", token_budget=500)
    assert isinstance(pack, ContextPack)
    assert len(pack.segments) >= 1  # at least one anchor returned


# --- S-08: port conformance ---

def test_s08_port():
    store = NetworkXStore()
    nodes = [Node(id="A", type=NodeType.ENTITY, label="Alpha")]
    store.upsert_nodes(nodes)
    mapping = {"Alpha": [1.0, 0.0]}
    embedder = _make_embedder(mapping)
    retriever = PathRetriever(store, embedder, nodes)
    assert isinstance(retriever, Retriever)


# --- S-09: overlay excluded ---

def test_s09_community_overlay_excluded():
    store = NetworkXStore()
    store.upsert_nodes([
        Node(id="A", type=NodeType.ENTITY, label="Alpha"),
        Node(id="C1", type=NodeType.COMMUNITY, label="Community1"),
        Node(id="B", type=NodeType.ENTITY, label="Beta"),
    ])
    store.upsert_edges([
        Edge(src="C1", dst="A", type=EdgeType.CONTAINS),
        Edge(src="C1", dst="B", type=EdgeType.CONTAINS),
    ])
    # Only path from A to B goes through C1 (community overlay) — must return None/fallback
    mapping = {"Alpha": [1.0, 0.0], "Beta": [0.9, 0.1], "Community1": [0.5, 0.5], "q": [1.0, 0.0]}
    embedder = _make_embedder(mapping)
    nodes = [Node(id="A", type=NodeType.ENTITY, label="Alpha"),
             Node(id="B", type=NodeType.ENTITY, label="Beta"),
             Node(id="C1", type=NodeType.COMMUNITY, label="Community1")]
    retriever = PathRetriever(store, embedder, nodes)
    pack = retriever.retrieve("q", token_budget=500)
    # Should not contain C1 in sources (path through overlay is excluded)
    assert "C1" not in {s.source for s in pack.segments}
```

- [ ] **Step 2: Rodar os testes para confirmar RED**

```bash
pytest tests/retrieval/test_path_retriever.py -v --no-header 2>&1 | tail -10
```
Esperado: `ImportError: cannot import name 'PathRetriever'`

- [ ] **Step 3: Implementar `path.py`**

```python
# glyph/retrieval/path.py
"""P8: PathRetriever — shortest-path arm.

For queries like 'what connects X to Y?' or decision-lineage questions.
Anchors to the two most similar nodes, finds shortest_path, renders the
walk as scored segments with edge-type annotations.
"""
from __future__ import annotations

from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, pack
from glyph.model.edge import EdgeType
from glyph.model.node import Node, NodeType
from glyph.store.port import GraphStore

_OVERLAY_NODE_TYPES = frozenset({NodeType.COMMUNITY})
_OVERLAY_EDGE_TYPES = frozenset({EdgeType.CONTAINS})


class PathRetriever:
    """Anchor query to two nodes; return the shortest structural path between them."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        nodes: Sequence[Node],
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._label: dict[str, str] = {}
        self._index = InMemoryVectorIndex()
        # Index only non-overlay nodes — community summaries are not path endpoints.
        indexable = [n for n in nodes if n.type not in _OVERLAY_NODE_TYPES]
        if indexable:
            ids = [n.id for n in indexable]
            vectors = embedder.embed([n.label for n in indexable])
            for nid, vec in zip(ids, vectors, strict=True):
                self._index.add(nid, vec)
            self._label = {n.id: n.label for n in indexable}

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        qvec = self._embedder.embed([query])[0]
        top2 = [key for key, _ in self._index.search(qvec, 2)]

        if len(top2) < 2:
            segments = [
                Segment(text=self._label.get(nid, nid), source=nid, score=1.0)
                for nid in top2
            ]
            return pack("graph", segments, token_budget)

        src, dst = top2[0], top2[1]
        path = self._store.shortest_path(
            src, dst,
            exclude_node_types=_OVERLAY_NODE_TYPES,
            exclude_edge_types=_OVERLAY_EDGE_TYPES,
        )

        if path is None:
            segments = [
                Segment(text=self._label.get(src, src), source=src, score=1.0),
                Segment(text=self._label.get(dst, dst), source=dst, score=1.0),
            ]
            return pack("graph", segments, token_budget)

        # Fetch edges between consecutive path nodes to annotate the walk.
        subgraph = self._store.subgraph(
            path.nodes, hops=0,
            exclude_node_types=_OVERLAY_NODE_TYPES,
            exclude_edge_types=_OVERLAY_EDGE_TYPES,
        )
        # Build adjacency: (src_id, dst_id) -> edge_type for quick lookup.
        adj: dict[tuple[str, str], str] = {
            (e.src, e.dst): e.type.value for e in subgraph.edges
        }

        n = len(path.nodes)
        segments: list[Segment] = []
        for i, nid in enumerate(path.nodes):
            label = self._label.get(nid, nid)
            # Annotate with the edge to the next node if present.
            if i < n - 1:
                next_id = path.nodes[i + 1]
                rel = adj.get((nid, next_id), "→")
                text = f"{label} —[{rel}]→"
            else:
                text = label
            score = 1.0 if i in (0, n - 1) else 0.7
            segments.append(Segment(text=text, source=nid, score=score))

        return pack("graph", segments, token_budget)
```

- [ ] **Step 4: Rodar testes para confirmar GREEN**

```bash
pytest tests/retrieval/test_path_retriever.py -v --no-header 2>&1 | tail -10
```

- [ ] **Step 5: Suite completa sem regressão**

```bash
pytest tests/ -q --no-header 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add glyph/retrieval/path.py tests/retrieval/test_path_retriever.py
git commit -m "feat(retrieval): PathRetriever — shortest-path arm for lineage queries"
```

---

## Task 3: MultiAnchorRetriever

**Files:**
- Create: `glyph/retrieval/multi_anchor.py`
- Create: `tests/retrieval/test_multi_anchor_retriever.py`

**Interfaces:**
- Consumes: `GraphStore.subgraph(seed: list[NodeId], hops, exclude_...) -> Subgraph`
- Produces: `MultiAnchorRetriever(store, embedder, nodes, hops=2, anchors=3)` com `retrieve(query, token_budget) -> ContextPack`

**Diferença-chave em relação ao `GraphRetriever`:** extração de entidades nomeadas da query (CamelCase, acrônimos, padrões `dec-`/`ADR-`) para garantir que cada entidade mencionada explicitamente na query ancora em um nó próprio — em vez de depender apenas do vetor da query completa (que pode favorecer uma entidade dominante).

**Heurística de extração:**
- Pattern: `r'\b[A-Z]\w{3,}\b'` — letra maiúscula + 3+ word chars (cobre CamelCase, acrônimos `ADR`, nomes próprios)
- Pattern: `r'(?:dec|adr)-[\w-]+'` com `re.IGNORECASE` — IDs de decisão
- Por cada entidade extraída: embed o termo, busca top-1 no índice
- Deduplica âncoras; se count < 2, completa com top-N por query embedding (fallback)

- [ ] **Step 1: Criar o arquivo de teste com os casos RED**

```python
# tests/retrieval/test_multi_anchor_retriever.py
"""S-10..S-13: MultiAnchorRetriever."""
import pytest
from unittest.mock import MagicMock
from glyph.model.contract import ContextPack
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.multi_anchor import MultiAnchorRetriever, _extract_named_entities
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


def _embedder(mapping: dict[str, list[float]]):
    emb = MagicMock()
    emb.embed.side_effect = lambda texts: [mapping[t] for t in texts]
    return emb


# --- S-10: entity extraction ---

def test_s10_camelcase_extracted():
    assert "AuthService" in _extract_named_entities("how does AuthService connect to PostgreSQL?")
    assert "PostgreSQL" in _extract_named_entities("how does AuthService connect to PostgreSQL?")


def test_s10_dec_id_extracted():
    assert "dec-121" in _extract_named_entities("what does dec-121 require?")


def test_s10_adr_id_extracted():
    assert "ADR-G7" in _extract_named_entities("show me ADR-G7 relations")


def test_s10_plain_query_returns_empty():
    # No named entities in an all-lowercase query
    result = _extract_named_entities("what is the main purpose of this system?")
    assert result == []


# --- S-11: multi-seed subgraph ---

def test_s11_multi_seed_finds_connection():
    """Query names two entities that are connected; the connection node appears in results."""
    store = NetworkXStore()
    nodes = [
        Node(id="A", type=NodeType.ENTITY, label="AuthService"),
        Node(id="B", type=NodeType.ENTITY, label="Connector"),
        Node(id="C", type=NodeType.ENTITY, label="PostgreSQL"),
    ]
    edges = [
        Edge(src="A", dst="B", type=EdgeType.REQUIRES),
        Edge(src="B", dst="C", type=EdgeType.REQUIRES),
    ]
    store.upsert_nodes(nodes)
    store.upsert_edges(edges)

    mapping = {
        "AuthService": [1.0, 0.0, 0.0],
        "Connector":   [0.0, 1.0, 0.0],
        "PostgreSQL":  [0.0, 0.0, 1.0],
        "AuthService": [1.0, 0.0, 0.0],  # entity embed
        "PostgreSQL":  [0.0, 0.0, 1.0],  # entity embed
        "how does AuthService connect to PostgreSQL?": [0.5, 0.0, 0.5],
    }
    # Deduplicate mapping keys (Python keeps last)
    m = {
        "AuthService": [1.0, 0.0, 0.0],
        "Connector":   [0.0, 1.0, 0.0],
        "PostgreSQL":  [0.0, 0.0, 1.0],
        "how does AuthService connect to PostgreSQL?": [0.5, 0.0, 0.5],
    }
    embedder = _embedder(m)
    retriever = MultiAnchorRetriever(store, embedder, nodes)
    pack = retriever.retrieve("how does AuthService connect to PostgreSQL?", token_budget=2000)
    sources = {s.source for s in pack.segments}
    # Connector (B) sits between A and C and should appear in the expanded subgraph
    assert "B" in sources


# --- S-12: fallback when no named entities ---

def test_s12_fallback_uses_embedding_anchors():
    store = NetworkXStore()
    nodes = [
        Node(id="X", type=NodeType.ENTITY, label="alpha"),
        Node(id="Y", type=NodeType.ENTITY, label="beta"),
    ]
    store.upsert_nodes(nodes)
    m = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0],
         "what is the system purpose?": [0.8, 0.2]}
    embedder = _embedder(m)
    retriever = MultiAnchorRetriever(store, embedder, nodes)
    pack = retriever.retrieve("what is the system purpose?", token_budget=500)
    assert isinstance(pack, ContextPack)
    assert len(pack.segments) >= 1


# --- S-13: port conformance ---

def test_s13_port():
    store = NetworkXStore()
    nodes = [Node(id="A", type=NodeType.ENTITY, label="alpha")]
    store.upsert_nodes(nodes)
    m = {"alpha": [1.0, 0.0]}
    embedder = _embedder(m)
    assert isinstance(MultiAnchorRetriever(store, embedder, nodes), Retriever)
```

- [ ] **Step 2: Rodar os testes para confirmar RED**

```bash
pytest tests/retrieval/test_multi_anchor_retriever.py -v --no-header 2>&1 | tail -10
```
Esperado: `ImportError: cannot import name 'MultiAnchorRetriever'`

- [ ] **Step 3: Implementar `multi_anchor.py`**

```python
# glyph/retrieval/multi_anchor.py
"""P9: MultiAnchorRetriever — entity-aware multi-seed expansion.

Extracts named entities from the query text (CamelCase, acronyms, dec-/ADR- IDs)
and anchors each to its closest node, ensuring every explicitly named entity
gets representation in the subgraph — unlike GraphRetriever which uses the
whole-query embedding and may miss entities dominated by others.

Falls back to top-N embedding anchors when the query contains no named entities.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, pack
from glyph.model.edge import EdgeType
from glyph.model.graph import Subgraph
from glyph.model.node import Node, NodeType
from glyph.store.port import GraphStore

_OVERLAY_NODE_TYPES = frozenset({NodeType.COMMUNITY})
_OVERLAY_EDGE_TYPES = frozenset({EdgeType.CONTAINS})

# CamelCase / PascalCase words (≥ 4 chars after the leading capital) OR
# decision/ADR ID patterns like dec-121, ADR-G7.
_CAMEL_RE = re.compile(r'\b[A-Z]\w{3,}\b')
_ID_RE = re.compile(r'(?:dec|adr)-[\w-]+', re.IGNORECASE)


def _extract_named_entities(query: str) -> list[str]:
    """Return distinct named entities found in query, preserving first-seen order."""
    seen: set[str] = set()
    entities: list[str] = []
    for m in _ID_RE.finditer(query):
        e = m.group(0)
        if e not in seen:
            seen.add(e)
            entities.append(e)
    for m in _CAMEL_RE.finditer(query):
        e = m.group(0)
        if e not in seen:
            seen.add(e)
            entities.append(e)
    return entities


class MultiAnchorRetriever:
    """Expand subgraph from all named entities in the query simultaneously."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        nodes: Sequence[Node],
        hops: int = 2,
        anchors: int = 3,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._hops = hops
        self._anchors = anchors
        self._label: dict[str, str] = {}
        self._index = InMemoryVectorIndex()
        indexable = [n for n in nodes if n.type not in _OVERLAY_NODE_TYPES]
        if indexable:
            ids = [n.id for n in indexable]
            vectors = embedder.embed([n.label for n in indexable])
            for nid, vec in zip(ids, vectors, strict=True):
                self._index.add(nid, vec)
            self._label = {n.id: n.label for n in indexable}

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        named = _extract_named_entities(query)
        seed: list[str] = []
        seen: set[str] = set()

        if named:
            # Anchor each named entity to its closest node.
            for entity in named:
                vec = self._embedder.embed([entity])[0]
                results = self._index.search(vec, 1)
                if results:
                    nid = results[0][0]
                    if nid not in seen:
                        seen.add(nid)
                        seed.append(nid)

        # Fallback: fill up to `anchors` with top-N by query embedding.
        if len(seed) < self._anchors:
            qvec = self._embedder.embed([query])[0]
            for nid, _ in self._index.search(qvec, self._anchors):
                if nid not in seen:
                    seen.add(nid)
                    seed.append(nid)

        if not seed:
            return pack("graph", [], token_budget)

        subgraph = self._store.subgraph(
            seed, self._hops,
            exclude_node_types=_OVERLAY_NODE_TYPES,
            exclude_edge_types=_OVERLAY_EDGE_TYPES,
        )
        return pack("graph", self._segments(subgraph, set(seed)), token_budget)

    def _segments(self, subgraph: Subgraph, anchors: set[str]) -> list[Segment]:
        label = {node.id: node.label for node in subgraph.nodes}
        out: dict[str, list[str]] = {}
        for edge in subgraph.edges:
            target = label.get(edge.dst, edge.dst)
            out.setdefault(edge.src, []).append(f"{edge.type.value} {target}")
        segments = []
        for node in subgraph.nodes:
            relations = "; ".join(out.get(node.id, []))
            text = f"{node.label} — {relations}" if relations else node.label
            score = 1.0 if node.id in anchors else 0.5
            segments.append(Segment(text=text, source=node.id, score=score))
        segments.sort(key=lambda s: (-s.score, s.source))
        return segments
```

- [ ] **Step 4: Rodar testes para confirmar GREEN**

```bash
pytest tests/retrieval/test_multi_anchor_retriever.py -v --no-header 2>&1 | tail -10
```

- [ ] **Step 5: Suite completa sem regressão**

```bash
pytest tests/ -q --no-header 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add glyph/retrieval/multi_anchor.py tests/retrieval/test_multi_anchor_retriever.py
git commit -m "feat(retrieval): MultiAnchorRetriever — entity-aware multi-seed subgraph expansion"
```

---

## Task 4: Exports e API pública

**Files:**
- Modify: `glyph/retrieval/__init__.py`
- Test: `tests/test_public_api.py` (já existe — adicionar verificações)

- [ ] **Step 1: Atualizar `glyph/retrieval/__init__.py`**

Ler o arquivo atual e adicionar os novos exports:

```python
# glyph/retrieval/__init__.py
"""Graph-aware retrieval and hybrid fusion."""
from glyph.retrieval.community import CommunityRetriever
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.hybrid import HybridRetriever
from glyph.retrieval.multi_anchor import MultiAnchorRetriever
from glyph.retrieval.path import PathRetriever
from glyph.retrieval.port import Retriever

__all__ = [
    "CommunityRetriever",
    "GraphRetriever",
    "HybridRetriever",
    "MultiAnchorRetriever",
    "PathRetriever",
    "Retriever",
]
```

- [ ] **Step 2: Verificar que o public API test passa**

```bash
pytest tests/test_public_api.py -v --no-header 2>&1 | tail -10
```

- [ ] **Step 3: Suite completa final**

```bash
pytest tests/ -q --no-header 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add glyph/retrieval/__init__.py
git commit -m "chore(retrieval): export PathRetriever and MultiAnchorRetriever"
```

---

## Self-Review

### Spec coverage

| Requisito | Task |
|---|---|
| Extração determinística de frontmatter (inline, block, scalar) | Task 1 S-01 |
| Extração de links Markdown locais | Task 1 S-02 |
| Nodes + edges válidos, sem duplicatas | Task 1 S-03 |
| Retorno vazio quando sem relações | Task 1 S-04 |
| Conformidade com Extractor port | Task 1 S-05 |
| PathRetriever: caminho encontrado | Task 2 S-06 |
| PathRetriever: fallback sem caminho | Task 2 S-07 |
| PathRetriever: Retriever port | Task 2 S-08 |
| PathRetriever: overlay excluído | Task 2 S-09 |
| MultiAnchorRetriever: extração de entidades | Task 3 S-10 |
| MultiAnchorRetriever: multi-seed | Task 3 S-11 |
| MultiAnchorRetriever: fallback embedding | Task 3 S-12 |
| MultiAnchorRetriever: Retriever port | Task 3 S-13 |
| Exports públicos | Task 4 |

### Placeholder scan

Nenhum placeholder encontrado — cada step tem código completo.

### Type consistency

- `PathRetriever` e `MultiAnchorRetriever` usam `InMemoryVectorIndex`, `Embedder`, `GraphStore`, `ContextPack`, `Segment`, `pack` — todos importados dos mesmos módulos que `GraphRetriever` usa.
- `MarkdownRelationExtractor.extract` retorna `tuple[Sequence[Node], Sequence[Edge]]` — conforme `Extractor` port.
- `_extract_named_entities` é exportada de `multi_anchor.py` e importada diretamente nos testes (S-10).

### TCM coverage

Todos os 13 Story IDs têm pelo menos um test ID mapeado. ✓
