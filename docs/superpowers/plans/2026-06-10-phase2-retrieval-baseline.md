# Phase 2 — Retrieval + Vector Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three retrieval arms — graph-aware, a fair vector baseline, and a hybrid — over the Monster Manual graph and its source chunks, all returning one comparable `ContextPack` (P2.1–P2.4). No measurement (that is Phase 3).

**Architecture:** A shared `glyph/embed/` infra (Embedder/VectorIndex ports + a local sentence-transformers adapter + a numpy in-memory cosine index). A `Segment`/`ContextPack` contract in `glyph/model/`. `glyph/baseline/vector.py` (VectorBaseline) embeds the same per-creature chunk texts; `glyph/retrieval/graph.py` (GraphRetriever) anchors a query against node labels and expands the neighborhood; `glyph/retrieval/hybrid.py` (HybridRetriever) fuses two injected `Retriever`s by reciprocal rank fusion. Layers stay decoupled: retrieval and baseline never import each other (hybrid takes injected retrievers).

**Tech Stack:** Python 3.11, Pydantic v2, NumPy (cosine index), sentence-transformers (local multilingual embeddings, lazily imported), pytest.

**Conventions:**
- TDD per step: failing test → watch it fail → minimal code → watch it pass → commit.
- Every commit ends with a second `-m` trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Run `python3 -m pytest`. If output looks empty/garbled (RTK proxy), re-run prefixed with `rtk proxy `. Gates: `ruff check glyph tests`, `ruff format --check glyph tests`, `mypy glyph`, coverage ≥ 90%.
- All retrieval logic is tested with a **deterministic fake embedder** — no model download. The real sentence-transformers adapter has a `@pytest.mark.slow` smoke test, deselected by default.

---

### Task 1: Dependencies, embed skeleton, pytest/CI config, ADR-G3

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Create: `glyph/embed/__init__.py`
- Create: `tests/embed/__init__.py` (empty)
- Create: `tests/retrieval/__init__.py` (empty)
- Create: `tests/baseline/__init__.py` (empty)
- Create: `docs/decisions/dec-g3-fair-vector-baseline.md`

- [ ] **Step 1: Add extras and the `slow` marker to `pyproject.toml`**

In `[project.optional-dependencies]`, after the `document` group add:
```toml
retrieval = [
    "numpy>=1.26",
]
embeddings = [
    "sentence-transformers>=3.0",
]
```
Replace the `[tool.pytest.ini_options]` block with:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-p no:asyncio --import-mode=importlib -m \"not live and not slow\" --cov=glyph --cov-report=term-missing --cov-fail-under=90"
markers = [
    "live: hits the live Anthropic API (requires ANTHROPIC_API_KEY); deselected by default",
    "slow: downloads a sentence-transformers model; deselected by default",
]
filterwarnings = [
    "ignore::DeprecationWarning:importlib._bootstrap",
    "ignore::DeprecationWarning:sys",
]
```
In `[[tool.mypy.overrides]]`, add a module entry for `sentence_transformers.*` (mirror the existing `networkx.*` / `fitz.*` overrides):
```toml
[[tool.mypy.overrides]]
module = ["sentence_transformers.*"]
ignore_missing_imports = true
```
(Keep the existing `networkx.*` and `fitz.*` overrides.)

- [ ] **Step 2: Install the retrieval extra**

Run: `python3 -m pip install -e ".[dev,document,retrieval]"`
Expected: `Successfully installed` (numpy present).

- [ ] **Step 3: Update CI to install the retrieval extra**

In `.github/workflows/ci.yml`, change the Install step to:
```yaml
      - name: Install
        run: pip install -e ".[dev,document,retrieval]"
```

- [ ] **Step 4: Create the embed package and test package markers**

Create `glyph/embed/__init__.py`:
```python
"""Embedding infrastructure: embedder + vector index ports and adapters."""
```
Create empty files `tests/embed/__init__.py`, `tests/retrieval/__init__.py`, `tests/baseline/__init__.py`.

- [ ] **Step 5: Write ADR-G3**

Create `docs/decisions/dec-g3-fair-vector-baseline.md`:
```markdown
# ADR-G3: Baseline vetorial justo e contrato de saída

**Data:** 2026-06-10
**Status:** Aceito

## Contexto

A tese do GLYPH é que retrieval graph-aware supera vector-only em queries que dependem de
relações entre entidades. O experimento só é válido se o baseline vetorial for forte e justo
— enfraquecê-lo invalida o número.

## Decisão

**Mesmo corpus:** o baseline vetorial indexa o texto dos mesmos chunks por criatura
(`chunk.by_creature` + `is_creature`) que geraram o grafo. O braço-grafo é a extração
estruturada desses chunks; o vetor é o texto cru deles embedado.

**Mesmo embedder e mesmo budget:** os dois braços usam o mesmo embedder local
(sentence-transformers multilingual) e cortam a saída no mesmo budget de token. O braço híbrido
funde os dois sob o mesmo budget.

**Contrato único:** `Segment`/`ContextPack` idêntico nos três modos, comparável token-a-token.

**Limitação declarada:** o budget é medido por estimativa de char nesta fase (não tokenizer
real). Declarado aqui e no benchmark (Fase 3).

## Consequências

**Positivas:** comparação controlada (mesma fonte, mesmo embedder, mesmo budget). O baseline é
implementação real (chunk + embedding + vector store + top-k), não espantalho.

**Trade-offs / a observar:** a estimativa de token por char é aproximada; a Fase 3 troca por
contagem real onde pesar. A fusão híbrida (reciprocal rank fusion) trata segmentos de grafo
(source = id de nó) e de vetor (source = label de chunk) como fontes distintas; unificar a
identidade entre representações fica para depois se o benchmark pedir.
```

- [ ] **Step 6: Verify the suite still passes**

Run: `python3 -m pytest -q`
Expected: `76 passed`, coverage gate met.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml glyph/embed/__init__.py tests/embed/__init__.py tests/retrieval/__init__.py tests/baseline/__init__.py docs/decisions/dec-g3-fair-vector-baseline.md
git commit -m "Add retrieval deps, embed skeleton, slow marker, and ADR-G3 (P2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Embed ports + in-memory cosine index

**Files:**
- Create: `glyph/embed/port.py`
- Create: `glyph/embed/memory_index.py`
- Test: `tests/embed/test_memory_index.py`

- [ ] **Step 1: Write the failing test**

Create `tests/embed/test_memory_index.py`:
```python
from glyph.embed.memory_index import InMemoryVectorIndex


def test_search_ranks_by_cosine_similarity() -> None:
    index = InMemoryVectorIndex()
    index.add("fire", [1.0, 0.0])
    index.add("cold", [0.0, 1.0])
    index.add("warm", [0.9, 0.1])
    results = index.search([1.0, 0.0], k=2)
    assert [key for key, _ in results] == ["fire", "warm"]
    assert results[0][1] > results[1][1]


def test_search_respects_k() -> None:
    index = InMemoryVectorIndex()
    for i in range(5):
        index.add(f"v{i}", [float(i), 1.0])
    assert len(index.search([1.0, 1.0], k=3)) == 3


def test_search_empty_index_returns_empty() -> None:
    assert InMemoryVectorIndex().search([1.0, 0.0], k=3) == []


def test_search_handles_zero_vector_without_error() -> None:
    index = InMemoryVectorIndex()
    index.add("z", [0.0, 0.0])
    results = index.search([1.0, 0.0], k=1)
    assert results[0][0] == "z"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/embed/test_memory_index.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.embed.memory_index'`.

- [ ] **Step 3: Implement `port.py` and `memory_index.py`**

Create `glyph/embed/port.py`:
```python
"""Embedding ports: turn text into vectors and search them by similarity."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = Sequence[float]


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Return one vector per input text."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    def add(self, key: str, vector: Vector) -> None:
        """Store a vector under ``key``."""
        ...

    def search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        """Return the ``k`` nearest ``(key, cosine_score)`` pairs, best first."""
        ...
```

Create `glyph/embed/memory_index.py`:
```python
"""In-memory cosine vector index (numpy). Zero servers; fits the target corpora."""

import numpy as np

from glyph.embed.port import Vector


class InMemoryVectorIndex:
    """A list-backed cosine index over float vectors."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._vectors: list[np.ndarray] = []

    def add(self, key: str, vector: Vector) -> None:
        self._keys.append(key)
        self._vectors.append(np.asarray(vector, dtype=np.float32))

    def search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        matrix = np.vstack(self._vectors)
        q = np.asarray(query, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        norms[norms == 0.0] = 1e-12
        scores = (matrix @ q) / norms
        order = np.argsort(-scores)[:k]
        return [(self._keys[i], float(scores[i])) for i in order]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/embed/test_memory_index.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full gates**

Run: `python3 -m mypy glyph` — expected clean. If mypy flags numpy expression types in `search` (e.g. `no-any-return` or operator types), add the minimal `# type: ignore[...]` only where mypy demands (the project sets `warn_unused_ignores`, so unused ignores fail). Re-run until clean.
Run: `python3 -m ruff check glyph tests` and `python3 -m ruff format --check glyph tests` — expected clean.

- [ ] **Step 6: Commit**

```bash
git add glyph/embed/port.py glyph/embed/memory_index.py tests/embed/test_memory_index.py
git commit -m "Add embedder/vector-index ports and in-memory cosine index (P2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: ContextPack contract

**Files:**
- Create: `glyph/model/contract.py`
- Test: `tests/model/test_contract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/model/test_contract.py`:
```python
from glyph.model.contract import ContextPack, Segment, estimate_tokens, pack


def test_estimate_tokens_is_a_ceil_char_quarter() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_pack_keeps_segments_within_budget() -> None:
    segments = [
        Segment(text="a" * 40, source="s1", score=0.9),  # 10 tokens
        Segment(text="b" * 40, source="s2", score=0.8),  # 10 tokens
        Segment(text="c" * 40, source="s3", score=0.7),  # 10 tokens
    ]
    result = pack("vector", segments, token_budget=20)
    assert [s.source for s in result.segments] == ["s1", "s2"]
    assert result.token_estimate == 20
    assert result.mode == "vector"


def test_pack_includes_first_segment_even_if_over_budget() -> None:
    segments = [Segment(text="x" * 400, source="big", score=1.0)]
    result = pack("graph", segments, token_budget=5)
    assert [s.source for s in result.segments] == ["big"]


def test_contextpack_round_trips_through_json() -> None:
    result = pack("hybrid", [Segment(text="hi", source="s", score=0.5)], token_budget=100)
    assert ContextPack.model_validate_json(result.model_dump_json()) == result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/model/test_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.model.contract'`.

- [ ] **Step 3: Implement `contract.py`**

Create `glyph/model/contract.py`:
```python
"""Unified retrieval output: a budget-bounded bundle of scored text segments."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

Mode = Literal["graph", "vector", "hybrid"]

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count by characters (declared limitation; real tokenizer is Phase 3)."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


class Segment(BaseModel):
    """One unit of retrieved context, with where it came from and how relevant it scored."""

    model_config = ConfigDict(frozen=True)

    text: str
    source: str
    score: float


class ContextPack(BaseModel):
    """The comparable output of every retrieval arm."""

    model_config = ConfigDict(frozen=True)

    mode: Mode
    segments: list[Segment]
    token_estimate: int


def pack(mode: Mode, segments: Sequence[Segment], token_budget: int) -> ContextPack:
    """Take score-ordered segments up to ``token_budget`` (always at least the first)."""
    chosen: list[Segment] = []
    total = 0
    for segment in segments:
        cost = estimate_tokens(segment.text)
        if chosen and total + cost > token_budget:
            break
        chosen.append(segment)
        total += cost
    return ContextPack(mode=mode, segments=chosen, token_estimate=total)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/model/test_contract.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/model/contract.py tests/model/test_contract.py
git commit -m "Add Segment/ContextPack contract with char-estimate budget (P2.4)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: VectorBaseline (P2.2)

**Files:**
- Create: `glyph/baseline/__init__.py`
- Create: `glyph/baseline/vector.py`
- Test: `tests/baseline/test_vector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/baseline/test_vector.py`:
```python
from collections.abc import Sequence

from glyph.baseline.vector import VectorBaseline
from glyph.embed.port import Vector


class _FakeEmbedder:
    """Deterministic 3-dim keyword embedder: [fogo, caverna, goblin]."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> Vector:
        low = text.lower()
        return [
            1.0 if "fogo" in low else 0.0,
            1.0 if "caverna" in low else 0.0,
            1.0 if "goblin" in low else 0.0,
        ]


def _docs() -> list[tuple[str, str]]:
    return [
        ("Goblin", "O goblin resiste a fogo."),
        ("Orc", "O orc habita cavernas."),
    ]


def test_retrieve_ranks_relevant_chunk_first() -> None:
    baseline = VectorBaseline(embedder=_FakeEmbedder())
    baseline.index(_docs())
    pack = baseline.retrieve("fogo", token_budget=1000)
    assert pack.mode == "vector"
    assert pack.segments[0].source == "Goblin"
    assert "fogo" in pack.segments[0].text


def test_retrieve_returns_segments_with_chunk_text_and_source() -> None:
    baseline = VectorBaseline(embedder=_FakeEmbedder())
    baseline.index(_docs())
    pack = baseline.retrieve("caverna", token_budget=1000)
    top = pack.segments[0]
    assert top.source == "Orc"
    assert top.text == "O orc habita cavernas."


def test_retrieve_on_empty_index_returns_empty_pack() -> None:
    pack = VectorBaseline(embedder=_FakeEmbedder()).retrieve("fogo", token_budget=1000)
    assert pack.segments == []
    assert pack.mode == "vector"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/baseline/test_vector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.baseline'`.

- [ ] **Step 3: Implement `baseline/vector.py`**

Create `glyph/baseline/__init__.py`:
```python
"""Vector baseline — the fair control arm of the benchmark."""
```

Create `glyph/baseline/vector.py`:
```python
"""P2.2: a fair vector baseline over the same per-creature chunk texts."""

from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder, VectorIndex
from glyph.model.contract import ContextPack, Segment, pack


class VectorBaseline:
    """Embed chunk texts, retrieve top-k by cosine, return a ContextPack."""

    def __init__(self, embedder: Embedder, index: VectorIndex | None = None) -> None:
        self._embedder = embedder
        self._index = index if index is not None else InMemoryVectorIndex()
        self._text: dict[str, str] = {}

    def index(self, documents: Sequence[tuple[str, str]]) -> None:
        """Index ``(source_label, text)`` documents (e.g. one per creature chunk)."""
        labels = [label for label, _ in documents]
        texts = [text for _, text in documents]
        for label, text, vector in zip(labels, texts, self._embedder.embed(texts), strict=True):
            self._index.add(label, vector)
            self._text[label] = text

    def retrieve(self, query: str, token_budget: int = 1000, k: int = 20) -> ContextPack:
        query_vector = self._embedder.embed([query])[0]
        hits = self._index.search(query_vector, k)
        segments = [
            Segment(text=self._text[key], source=key, score=score) for key, score in hits
        ]
        return pack("vector", segments, token_budget)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/baseline/test_vector.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/baseline/__init__.py glyph/baseline/vector.py tests/baseline/test_vector.py
git commit -m "Add VectorBaseline over shared chunk texts (P2.2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: GraphRetriever (P2.1)

**Files:**
- Create: `glyph/retrieval/__init__.py`
- Create: `glyph/retrieval/port.py`
- Create: `glyph/retrieval/graph.py`
- Test: `tests/retrieval/test_graph.py`

- [ ] **Step 1: Write the failing test**

Create `tests/retrieval/test_graph.py`:
```python
from collections.abc import Sequence

from glyph.embed.port import Vector
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore


class _FakeEmbedder:
    """Deterministic 3-dim keyword embedder: [fogo, caverna, goblin]."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> Vector:
        low = text.lower()
        return [
            1.0 if "fogo" in low else 0.0,
            1.0 if "caverna" in low else 0.0,
            1.0 if "goblin" in low else 0.0,
        ]


def _nodes() -> list[Node]:
    return [
        Node(id="goblin", type=NodeType.ENTITY, label="Goblin"),
        Node(id="fogo", type=NodeType.CONCEPT, label="fogo"),
        Node(id="caverna", type=NodeType.CONCEPT, label="caverna"),
    ]


def _store(nodes: list[Node]) -> NetworkXStore:
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([Edge(src="goblin", dst="fogo", type=EdgeType.RESISTS)])
    return store


def test_graph_retriever_satisfies_the_port() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes)
    assert isinstance(retriever, Retriever)


def test_retrieve_anchors_query_and_returns_neighborhood() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(
        store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes, hops=1
    )
    result = retriever.retrieve("fogo", token_budget=1000)
    assert result.mode == "graph"
    sources = {s.source for s in result.segments}
    assert "fogo" in sources  # anchored node
    assert "goblin" in sources  # its neighbor (resists fogo)


def test_segment_text_includes_relations() -> None:
    nodes = _nodes()
    retriever = GraphRetriever(
        store=_store(nodes), embedder=_FakeEmbedder(), nodes=nodes, hops=1
    )
    result = retriever.retrieve("goblin", token_budget=1000)
    goblin = next(s for s in result.segments if s.source == "goblin")
    assert "resists" in goblin.text
    assert "fogo" in goblin.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/retrieval/test_graph.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.retrieval'`.

- [ ] **Step 3: Implement `retrieval/port.py` and `retrieval/graph.py`**

Create `glyph/retrieval/__init__.py`:
```python
"""Graph-aware retrieval and hybrid fusion."""
```

Create `glyph/retrieval/port.py`:
```python
"""The Retriever port: every arm turns a query into a comparable ContextPack."""

from typing import Protocol, runtime_checkable

from glyph.model.contract import ContextPack


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, token_budget: int) -> ContextPack:
        """Return retrieved context for ``query`` within ``token_budget``."""
        ...
```

Create `glyph/retrieval/graph.py`:
```python
"""P2.1: graph-aware retrieval — anchor a query, expand the neighborhood."""

from collections.abc import Sequence

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder
from glyph.model.contract import ContextPack, Segment, pack
from glyph.model.graph import Subgraph
from glyph.model.node import Node
from glyph.store.port import GraphStore


class GraphRetriever:
    """Embed node labels once; anchor a query to the nearest labels and expand by hops."""

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
        self._label = {node.id: node.label for node in nodes}
        self._index = InMemoryVectorIndex()
        ids = list(self._label)
        vectors = embedder.embed([self._label[node_id] for node_id in ids])
        for node_id, vector in zip(ids, vectors, strict=True):
            self._index.add(node_id, vector)

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        query_vector = self._embedder.embed([query])[0]
        anchors = [key for key, _ in self._index.search(query_vector, self._anchors)]
        subgraph = self._store.subgraph(anchors, self._hops)
        return pack("graph", self._segments(subgraph, set(anchors)), token_budget)

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
        segments.sort(key=lambda s: -s.score)
        return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/retrieval/test_graph.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/retrieval/__init__.py glyph/retrieval/port.py glyph/retrieval/graph.py tests/retrieval/test_graph.py
git commit -m "Add graph-aware retrieval with embedding anchoring (P2.1)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: HybridRetriever (P2.3)

**Files:**
- Create: `glyph/retrieval/hybrid.py`
- Test: `tests/retrieval/test_hybrid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/retrieval/test_hybrid.py`:
```python
from glyph.model.contract import ContextPack, Segment
from glyph.retrieval.hybrid import HybridRetriever
from glyph.retrieval.port import Retriever


class _CannedRetriever:
    def __init__(self, mode, segments: list[Segment]) -> None:
        self._mode = mode
        self._segments = segments

    def retrieve(self, query: str, token_budget: int) -> ContextPack:
        return ContextPack(mode=self._mode, segments=self._segments, token_estimate=0)


def test_hybrid_satisfies_the_port() -> None:
    graph = _CannedRetriever("graph", [])
    vector = _CannedRetriever("vector", [])
    assert isinstance(HybridRetriever(graph, vector), Retriever)


def test_hybrid_fuses_and_dedupes_by_source() -> None:
    graph = _CannedRetriever(
        "graph",
        [Segment(text="A", source="a", score=1.0), Segment(text="B", source="b", score=0.9)],
    )
    vector = _CannedRetriever(
        "vector",
        [Segment(text="B", source="b", score=0.8), Segment(text="C", source="c", score=0.7)],
    )
    result = HybridRetriever(graph, vector).retrieve("q", token_budget=1000)
    assert result.mode == "hybrid"
    sources = [s.source for s in result.segments]
    assert sources.count("b") == 1  # deduped across arms
    assert set(sources) == {"a", "b", "c"}
    assert sources[0] == "b"  # appears in both arms -> highest fused rank


def test_hybrid_respects_token_budget() -> None:
    graph = _CannedRetriever("graph", [Segment(text="x" * 40, source="g", score=1.0)])
    vector = _CannedRetriever("vector", [Segment(text="y" * 40, source="v", score=1.0)])
    result = HybridRetriever(graph, vector).retrieve("q", token_budget=10)
    assert len(result.segments) == 1  # only one 10-token segment fits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/retrieval/test_hybrid.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.retrieval.hybrid'`.

- [ ] **Step 3: Implement `retrieval/hybrid.py`**

Create `glyph/retrieval/hybrid.py`:
```python
"""P2.3: hybrid retrieval — fuse two retrievers by reciprocal rank fusion."""

from glyph.model.contract import ContextPack, Segment, pack
from glyph.retrieval.port import Retriever

_RRF_K = 60


class HybridRetriever:
    """Run a graph and a vector retriever and fuse their segments (injected, not imported)."""

    def __init__(self, graph: Retriever, vector: Retriever, rrf_k: int = _RRF_K) -> None:
        self._graph = graph
        self._vector = vector
        self._rrf_k = rrf_k

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        graph_pack = self._graph.retrieve(query, token_budget)
        vector_pack = self._vector.retrieve(query, token_budget)
        fused = self._fuse(graph_pack.segments, vector_pack.segments)
        return pack("hybrid", fused, token_budget)

    def _fuse(self, *arms: list[Segment]) -> list[Segment]:
        scores: dict[str, float] = {}
        first_seen: dict[str, Segment] = {}
        for segments in arms:
            for rank, segment in enumerate(segments):
                scores[segment.source] = scores.get(segment.source, 0.0) + 1.0 / (
                    self._rrf_k + rank + 1
                )
                first_seen.setdefault(segment.source, segment)
        merged = [
            Segment(text=first_seen[source].text, source=source, score=score)
            for source, score in scores.items()
        ]
        merged.sort(key=lambda s: -s.score)
        return merged
```

Note: `_fuse(self, *arms)` — call it `self._fuse(graph_pack.segments, vector_pack.segments)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/retrieval/test_hybrid.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add glyph/retrieval/hybrid.py tests/retrieval/test_hybrid.py
git commit -m "Add hybrid retrieval via reciprocal rank fusion (P2.3)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: SentenceTransformerEmbedder (local adapter)

**Files:**
- Create: `glyph/embed/sentence_transformers_embedder.py`
- Modify: `glyph/embed/__init__.py`
- Test: `tests/embed/test_sentence_transformers_embedder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/embed/test_sentence_transformers_embedder.py`:
```python
import importlib.util

import pytest

from glyph.embed.port import Embedder
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder


class _FakeModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]


def test_embedder_delegates_to_the_model_and_satisfies_the_port() -> None:
    embedder = SentenceTransformerEmbedder(model=_FakeModel())
    assert isinstance(embedder, Embedder)
    vectors = embedder.embed(["a", "abc"])
    assert vectors == [[1.0, 1.0], [3.0, 1.0]]


@pytest.mark.slow
def test_real_model_ranks_similar_text_higher() -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")
    from glyph.embed.memory_index import InMemoryVectorIndex

    embedder = SentenceTransformerEmbedder()
    index = InMemoryVectorIndex()
    for key, text in [("fogo", "resistência a fogo"), ("frio", "imunidade a frio")]:
        index.add(key, embedder.embed([text])[0])
    top = index.search(embedder.embed(["dano de fogo"])[0], k=1)
    assert top[0][0] == "fogo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/embed/test_sentence_transformers_embedder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'glyph.embed.sentence_transformers_embedder'`. (The slow test is deselected.)

- [ ] **Step 3: Implement the adapter**

Create `glyph/embed/sentence_transformers_embedder.py`:
```python
"""Local multilingual embedder (sentence-transformers), lazily imported."""

from collections.abc import Sequence

from glyph.embed.port import Vector

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbedder:
    """Wrap a SentenceTransformer model behind the Embedder port."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, model: object | None = None) -> None:
        if model is None:  # pragma: no cover - downloads weights, exercised by the slow test
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors = self._model.encode(list(texts))  # type: ignore[attr-defined]
        return [list(map(float, vector)) for vector in vectors]
```

- [ ] **Step 4: Export the adapter from the package**

Replace `glyph/embed/__init__.py` with:
```python
"""Embedding infrastructure: embedder + vector index ports and adapters."""

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder, VectorIndex
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder

__all__ = ["Embedder", "InMemoryVectorIndex", "SentenceTransformerEmbedder", "VectorIndex"]
```

- [ ] **Step 5: Run tests and gates**

Run: `python3 -m pytest tests/embed/ -q` — expected PASS (the fake-model test runs; the slow test is deselected).
Run: `python3 -m pytest -m slow tests/embed/test_sentence_transformers_embedder.py --collect-only -q` — expected: collects 1 test (do NOT run it; it downloads a model).
Run: `python3 -m mypy glyph`, `python3 -m ruff check glyph tests`, `python3 -m ruff format --check glyph tests` — expected clean. Report whether the `# type: ignore[attr-defined]` on `self._model.encode` was needed (it should be, since `model` is typed `object`).

- [ ] **Step 6: Commit**

```bash
git add glyph/embed/sentence_transformers_embedder.py glyph/embed/__init__.py tests/embed/test_sentence_transformers_embedder.py
git commit -m "Add local sentence-transformers embedder adapter (P2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Architecture invariants for the new layers

**Files:**
- Modify: `tests/architecture/test_dependencies.py`

- [ ] **Step 1: Write the failing tests**

In `tests/architecture/test_dependencies.py`, add `"glyph.baseline"` and `"glyph.retrieval"` and `"glyph.embed"` to the `INWARD_LAYERS` tuple (so `model` may not import them), and append these test functions:
```python
def test_embed_imports_only_external_and_its_own_package() -> None:
    forbidden = ("glyph.model", "glyph.store", "glyph.extract", "glyph.retrieval", "glyph.baseline")
    for pyfile in _modules_under("embed"):
        offenders = {m for m in _imported_modules(pyfile) if m.startswith(forbidden)}
        assert not offenders, f"{pyfile.name} imports {offenders}"


def test_retrieval_does_not_import_baseline() -> None:
    for pyfile in _modules_under("retrieval"):
        offenders = {m for m in _imported_modules(pyfile) if m.startswith("glyph.baseline")}
        assert not offenders, f"{pyfile.name} imports baseline: {offenders}"


def test_baseline_does_not_import_retrieval() -> None:
    for pyfile in _modules_under("baseline"):
        offenders = {m for m in _imported_modules(pyfile) if m.startswith("glyph.retrieval")}
        assert not offenders, f"{pyfile.name} imports retrieval: {offenders}"
```
Also update `test_model_does_not_import_outer_layers` — it already iterates `INWARD_LAYERS`, so adding the three layers there is the only change needed for the model check.

- [ ] **Step 2: Run tests to verify they pass (these guard, so they pass once correct)**

Run: `python3 -m pytest tests/architecture/ -q`
Expected: PASS. Then, to prove the guard works, temporarily add `import glyph.baseline` to the top of `glyph/retrieval/hybrid.py`, run `python3 -m pytest tests/architecture/test_dependencies.py::test_retrieval_does_not_import_baseline -q` (expected FAIL), then remove the import and re-run (expected PASS).

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS, coverage ≥ 90%.

- [ ] **Step 4: Commit**

```bash
git add tests/architecture/test_dependencies.py
git commit -m "Extend architecture invariants to embed/retrieval/baseline (P2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Wiring demo over the real Monster Manual graph

The demo is the composition root that joins store + embed + the three retrievers. It lives in `scripts/` (outside the package) and is the only place importing both `glyph.retrieval` and `glyph.baseline`. It uses the real local embedder (no API cost) and requires `pip install -e ".[document,retrieval,embeddings]"`.

**Files:**
- Create: `scripts/retrieve_demo.py`

- [ ] **Step 1: Implement the demo**

Create `scripts/retrieve_demo.py`:
```python
"""Run the three retrieval arms over the Monster Manual graph for manual inspection.

Usage:
    python3 scripts/retrieve_demo.py out/monster-manual.json "<Monster Manual.pdf>" "que criaturas resistem a fogo?"

Uses local sentence-transformers embeddings (no API cost). Requires:
    pip install -e ".[document,retrieval,embeddings]"
"""

import sys

from glyph.baseline.vector import VectorBaseline
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder
from glyph.extract.document import chunk, pdf
from glyph.model.contract import ContextPack
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.hybrid import HybridRetriever
from glyph.store.networkx_store import NetworkXStore


def _show(name: str, result: ContextPack) -> None:
    print(f"\n=== {name} ({result.token_estimate} tok, {len(result.segments)} segments) ===")
    for segment in result.segments[:8]:
        print(f"  [{segment.score:.3f}] {segment.source}: {segment.text[:110]}")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    graph_path, book_path, query = argv[1], argv[2], argv[3]

    store = NetworkXStore.load(graph_path)
    nodes = store.subgraph([n for n in _all_ids(graph_path)], hops=0).nodes

    documents = [
        (piece.label, piece.text)
        for piece in chunk.by_creature(pdf.load(book_path))
        if chunk.is_creature(piece)
    ]

    embedder = SentenceTransformerEmbedder()
    graph = GraphRetriever(store=store, embedder=embedder, nodes=nodes)
    vector = VectorBaseline(embedder=embedder)
    vector.index(documents)
    hybrid = HybridRetriever(graph, vector)

    print(f"query: {query!r}")
    _show("graph", graph.retrieve(query))
    _show("vector", vector.retrieve(query))
    _show("hybrid", hybrid.retrieve(query))
    return 0


def _all_ids(graph_path: str) -> list[str]:
    import json
    from pathlib import Path

    payload = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    return [node["id"] for node in payload["nodes"]]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Verify the demo wires up without the model (no-arg path)**

Run: `python3 scripts/retrieve_demo.py; echo "exit=$?"`
Expected: prints the usage docstring, `exit=2`. (No model download on the no-arg path — it returns before constructing the embedder.)
Run: `python3 -m ruff check scripts` (informational) — expected clean.

- [ ] **Step 3: Run the full gates one last time**

Run: `python3 -m pytest -q` (PASS, coverage ≥ 90%), `python3 -m mypy glyph` (clean), `python3 -m ruff check glyph tests` (clean), `python3 -m ruff format --check glyph tests` (clean).

- [ ] **Step 4: Commit**

```bash
git add scripts/retrieve_demo.py
git commit -m "Add retrieval demo wiring over the Monster Manual graph (P2)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Manual validation (optional, local, no API cost)**

With `pip install -e ".[document,retrieval,embeddings]"`, run:
```bash
python3 scripts/retrieve_demo.py out/monster-manual.json "$HOME/Downloads/3-Monster Manual.pdf" "que criaturas resistem a fogo?"
```
Expected: three ContextPacks printed. Eyeball that the graph arm surfaces creatures connected to "fogo" and the vector arm surfaces fogo-mentioning chunks. This downloads the embedding model on first run (~470 MB, local, free). Record nothing to git — this is inspection only.

---

## Self-Review

**Spec coverage:**
- P2.1 graph retrieval → Task 5. ✓
- P2.2 vector baseline → Task 4. ✓
- P2.3 hybrid → Task 6. ✓
- P2.4 ContextPack contract → Task 3. ✓
- Embeddings (local ST + in-memory index) → Tasks 2, 7. ✓
- ADR-G3 + fair-baseline (same chunks, same budget) → Tasks 1, 4. ✓
- Layer decoupling (Retriever protocol, hybrid by injection) → Tasks 5, 6, 8. ✓
- Architecture invariants extended → Task 8. ✓
- Slow ST test deselected; CI light (numpy only) → Tasks 1, 7. ✓
- Manual validation over real MM → Task 9. ✓

**Placeholder scan:** every code step shows complete code; commands have expected output; no TBD/TODO.

**Type consistency:** `Vector = Sequence[float]`; `Embedder.embed(texts) -> list[Vector]`; `VectorIndex.add(key, vector)` / `search(query, k) -> list[tuple[str, float]]`; `Segment{text, source, score}`; `ContextPack{mode, segments, token_estimate}`; `pack(mode, segments, token_budget)`; `estimate_tokens(text) -> int`; `VectorBaseline.index(documents: Sequence[tuple[str,str]])` / `retrieve(query, token_budget, k)`; `GraphRetriever(store, embedder, nodes, hops, anchors)` / `retrieve(query, token_budget)`; `HybridRetriever(graph, vector, rrf_k)` / `retrieve(query, token_budget)`; `Retriever.retrieve(query, token_budget)`. Names are consistent across tasks.

**Coverage note:** the `SentenceTransformerEmbedder` real-model branch (`model is None`) is `# pragma: no cover` (covered by the slow test). The `scripts/` demo is outside `--cov=glyph`. All retrieval/baseline/embed logic is covered by fake-embedder tests.
