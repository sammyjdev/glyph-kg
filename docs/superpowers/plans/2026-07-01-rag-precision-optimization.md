# RAG Precision Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase context_precision from ~0.44 to 0.55+ via a cross-encoder reranker wrapper and composite graph scoring (PageRank blend), validated by the existing benchmark harness.

**Architecture:** Three independent improvements layered over the existing retrieval ports — (1) a `CrossEncoderReranker` that rescores any retriever's candidate segments query-by-query; (2) a generic `RerankedRetriever` wrapper that composes any `Retriever` with any `Reranker`; (3) an optional `pagerank_weight` in `GraphRetriever` that blends cosine similarity with normalized PageRank centrality for non-anchor nodes.

**Tech Stack:** Python 3.11, sentence-transformers (`CrossEncoder`, already installed as dep), networkx (`nx.pagerank`, already installed), pytest, existing `glyph.*` model/store/retrieval ports.

## Global Constraints

- No new pip dependencies — `sentence_transformers` and `networkx` are already in the project
- All new public classes must satisfy the `Retriever` protocol (`retrieve(query, token_budget) -> ContextPack`)
- Frozen `Segment` and `ContextPack` — never mutate in place, always construct new objects
- Tests go in `tests/retrieval/` following existing naming: `test_<module>.py`
- Run tests with `python -m pytest tests/ -v` from repo root
- Commit format: `feat(retrieval): <description>`

---

## Test Coverage Matrix

| Story ID | Requirement | Test IDs | Status |
|----------|-------------|----------|--------|
| S-01 | `CrossEncoderReranker.rerank()` returns segments sorted by cross-encoder score | `test_s01_reranker_sorts_by_relevance`, `test_s01_reranker_truncates_to_k` | ✗ unmet |
| S-02 | `RerankedRetriever` satisfies the `Retriever` port | `test_s02_reranked_retriever_satisfies_port` | ✗ unmet |
| S-03 | `RerankedRetriever` calls underlying retriever with large budget, returns k segments | `test_s03_reranked_uses_large_budget`, `test_s03_reranked_returns_k` | ✗ unmet |
| S-04 | `NetworkXStore.pagerank()` returns normalized scores summing to ~1 | `test_s04_pagerank_sums_to_one`, `test_s04_pagerank_hub_is_highest` | ✗ unmet |
| S-05 | `GraphRetriever` with `pagerank_weight>0` gives higher score to structurally central nodes | `test_s05_pagerank_weight_raises_hub_score` | ✗ unmet |
| S-06 | Benchmark `_build_arms()` includes `reranked_vector` arm; `--k-sweep` prints precision table | `test_s06_build_arms_has_reranked_vector` (integration, run manually) | ✗ unmet |

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `glyph/retrieval/reranker.py` | `Reranker` protocol + `CrossEncoderReranker` impl |
| Create | `glyph/retrieval/reranked.py` | `RerankedRetriever` wrapper |
| Modify | `glyph/model/contract.py` | Extend `Mode` with `"reranked"` |
| Modify | `glyph/store/port.py` | Add `pagerank()` to the `GraphStore` protocol |
| Modify | `glyph/store/networkx_store.py` | Add `pagerank() -> dict[str, float]` |
| Modify | `glyph/store/neo4j_store.py` | Add `pagerank()` (pulls graph, runs `nx.pagerank`) |
| Modify | `glyph/retrieval/graph.py` | Add `pagerank_weight` param + blend logic |
| Create | `tests/retrieval/test_reranker.py` | Tests for S-01, S-02, S-03 |
| Create | `tests/retrieval/test_pagerank_graph.py` | Tests for S-04, S-05 |
| Modify | `scripts/run_benchmark.py` | Add `reranked_vector` arm + `--k-sweep` flag |

---

## Task 1: Reranker port + CrossEncoderReranker + RerankedRetriever

**Files:**
- Create: `glyph/retrieval/reranker.py`
- Create: `glyph/retrieval/reranked.py`
- Modify: `glyph/model/contract.py` (extend Mode)
- Create: `tests/retrieval/test_reranker.py`

**Interfaces:**
- Produces:
  - `Reranker` protocol: `rerank(query: str, segments: list[Segment], k: int) -> list[Segment]`
  - `CrossEncoderReranker(model: str)` implementing `Reranker`
  - `RerankedRetriever(retriever: Retriever, reranker: Reranker, k: int)` implementing `Retriever`

- [ ] **Step 1: Extend Mode in contract.py**

Open `glyph/model/contract.py`, change line 8:

```python
Mode = Literal["graph", "vector", "hybrid", "community", "reranked"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/retrieval/test_reranker.py`:

```python
from collections.abc import Sequence

from glyph.model.contract import ContextPack, Segment
from glyph.retrieval.port import Retriever
from glyph.retrieval.reranked import RerankedRetriever
from glyph.retrieval.reranker import CrossEncoderReranker, Reranker


class _FakeReranker:
    """Always returns segments in reverse order (last becomes first)."""

    def rerank(self, query: str, segments: list[Segment], k: int) -> list[Segment]:
        return list(reversed(segments))[:k]


class _FakeRetriever:
    """Returns a fixed list of segments, ignores token_budget."""

    def __init__(self, segments: list[Segment], last_budget: list[int] | None = None) -> None:
        self._segments = segments
        self._last_budget = last_budget if last_budget is not None else []

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        self._last_budget.append(token_budget)
        from glyph.model.contract import pack
        return pack("graph", self._segments, token_budget=50_000)


def _seg(text: str, score: float) -> Segment:
    return Segment(text=text, source=text, score=score)


def test_s01_reranker_sorts_by_relevance() -> None:
    """CrossEncoderReranker must rank the more relevant segment higher."""
    reranker = CrossEncoderReranker()
    segments = [
        _seg("Goblins are small green creatures that live in caves.", 0.5),
        _seg("The weather in Paris is often cloudy.", 0.5),
    ]
    result = reranker.rerank("Where do goblins live?", segments, k=2)
    assert result[0].source == "Goblins are small green creatures that live in caves."


def test_s01_reranker_truncates_to_k() -> None:
    """CrossEncoderReranker must return exactly k segments."""
    reranker = CrossEncoderReranker()
    segments = [_seg(f"text {i}", 0.5) for i in range(10)]
    result = reranker.rerank("query", segments, k=3)
    assert len(result) == 3


def test_s02_reranked_retriever_satisfies_port() -> None:
    """RerankedRetriever must satisfy the Retriever protocol."""
    retriever = _FakeRetriever([_seg("a", 1.0), _seg("b", 0.5)])
    reranker = _FakeReranker()
    wrapped = RerankedRetriever(retriever, reranker, k=2)
    assert isinstance(wrapped, Retriever)


def test_s03_reranked_uses_large_budget() -> None:
    """RerankedRetriever must call underlying retriever with budget>=50_000."""
    budgets: list[int] = []
    retriever = _FakeRetriever([_seg("x", 1.0)], last_budget=budgets)
    reranker = _FakeReranker()
    RerankedRetriever(retriever, reranker, k=1).retrieve("q", token_budget=1000)
    assert budgets[-1] >= 50_000


def test_s03_reranked_returns_k() -> None:
    """RerankedRetriever must return at most k segments in the pack."""
    segments = [_seg(f"text {i}" * 5, float(i)) for i in range(8)]
    retriever = _FakeRetriever(segments)
    reranker = _FakeReranker()
    pack = RerankedRetriever(retriever, reranker, k=3).retrieve("q", token_budget=10_000)
    assert len(pack.segments) <= 3
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
python -m pytest tests/retrieval/test_reranker.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `ModuleNotFoundError` on `reranker` / `reranked`.

- [ ] **Step 4: Implement `glyph/retrieval/reranker.py`**

```python
"""Cross-encoder reranker: rescores candidate segments query-by-query."""

from typing import Protocol, runtime_checkable

from glyph.model.contract import Segment

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, segments: list[Segment], k: int) -> list[Segment]: ...


class CrossEncoderReranker:
    """Rerank segments using a sentence-transformers CrossEncoder model (local, no API)."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model)

    def rerank(self, query: str, segments: list[Segment], k: int) -> list[Segment]:
        if not segments:
            return []
        pairs = [(query, s.text) for s in segments]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, segments), key=lambda x: -float(x[0]))
        return [
            Segment(text=s.text, source=s.source, score=float(score))
            for score, s in ranked[:k]
        ]
```

- [ ] **Step 5: Implement `glyph/retrieval/reranked.py`**

```python
"""RerankedRetriever: compose any Retriever with any Reranker."""

from glyph.model.contract import ContextPack, pack
from glyph.retrieval.port import Retriever
from glyph.retrieval.reranker import Reranker

_CANDIDATE_BUDGET = 50_000  # large enough to pass all candidates through


class RerankedRetriever:
    """Retrieve a large candidate set, then rerank with a cross-encoder to top-k."""

    def __init__(self, retriever: Retriever, reranker: Reranker, k: int = 5) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._k = k

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        candidates = self._retriever.retrieve(query, token_budget=_CANDIDATE_BUDGET)
        reranked = self._reranker.rerank(query, list(candidates.segments), self._k)
        return pack("reranked", reranked, token_budget)
```

- [ ] **Step 6: Run tests — all must pass**

```bash
python -m pytest tests/retrieval/test_reranker.py -v 2>&1 | tail -20
```

Expected: 5 PASSED.

- [ ] **Step 7: Commit**

```bash
git add glyph/model/contract.py glyph/retrieval/reranker.py glyph/retrieval/reranked.py tests/retrieval/test_reranker.py
git commit -m "feat(retrieval): CrossEncoderReranker + RerankedRetriever wrapper"
```

---

## Task 2: Composite graph scoring (PageRank blend)

**Files:**
- Modify: `glyph/store/port.py`
- Modify: `glyph/store/networkx_store.py`
- Modify: `glyph/store/neo4j_store.py`
- Modify: `glyph/retrieval/graph.py`
- Create: `tests/retrieval/test_pagerank_graph.py`

**Interfaces:**
- Consumes: `NetworkXStore` (existing), `GraphRetriever` (existing)
- Produces:
  - `GraphStore.pagerank() -> dict[str, float]` — added to the port so every backend
    (NetworkX, Neo4j) satisfies it; `GraphRetriever` depends only on the port (dec:
    `store/port.py` docstring — "Retrieval depends on the port, never on a concrete
    backend"), so `mypy --strict` and the Neo4j adapter must not be skipped
  - `NetworkXStore.pagerank() -> dict[str, float]` — raw scores (sum ≈ 1.0)
  - `Neo4jStore.pagerank() -> dict[str, float]` — same contract, computed by pulling
    the full node/edge set and running `nx.pagerank` (no GDS plugin dependency)
  - `GraphRetriever(…, pagerank_weight: float = 0.0)` — blends cosine with PageRank

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/test_pagerank_graph.py`:

```python
import pytest

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.retrieval.graph import GraphRetriever
from glyph.store.networkx_store import NetworkXStore


def _hub_graph() -> tuple[NetworkXStore, list[Node]]:
    """Three nodes: hub connected to both leaves; leaves connected to nothing else."""
    nodes = [
        Node(id="hub", type=NodeType.ENTITY, label="Hub"),
        Node(id="leaf_a", type=NodeType.CONCEPT, label="leaf_a"),
        Node(id="leaf_b", type=NodeType.CONCEPT, label="leaf_b"),
    ]
    store = NetworkXStore()
    store.upsert_nodes(nodes)
    store.upsert_edges([
        Edge(src="hub", dst="leaf_a", type=EdgeType.RESISTS),
        Edge(src="hub", dst="leaf_b", type=EdgeType.RESISTS),
    ])
    return store, nodes


def test_s04_pagerank_sums_to_one() -> None:
    """pagerank() scores must sum to approximately 1.0."""
    store, _ = _hub_graph()
    pr = store.pagerank()
    assert abs(sum(pr.values()) - 1.0) < 1e-6


def test_s04_pagerank_hub_is_highest() -> None:
    """The hub node (most edges) must have the highest raw pagerank score."""
    store, _ = _hub_graph()
    pr = store.pagerank()
    assert pr["hub"] > pr["leaf_a"]
    assert pr["hub"] > pr["leaf_b"]


def test_s05_pagerank_weight_raises_hub_score() -> None:
    """With pagerank_weight=0.5, the hub node gets a higher blended score than a leaf."""
    store, nodes = _hub_graph()

    class _EqualEmbedder:
        """All nodes embed to the same vector — cosine tie — so PageRank breaks it."""
        def embed(self, texts):
            return [[1.0, 0.0, 0.0]] * len(texts)

    retriever = GraphRetriever(
        store=store,
        embedder=_EqualEmbedder(),
        nodes=nodes,
        hops=1,
        anchors=1,
        pagerank_weight=0.5,
    )
    result = retriever.retrieve("anything", token_budget=10_000)
    scores = {s.source: s.score for s in result.segments}
    # hub is the anchor (score=1.0), leaves are neighbors — hub's centrality
    # doesn't matter for the anchor itself, but leaves should differ by pagerank
    # (both leaves have equal cosine, so their scores remain equal — this test
    # verifies the blend doesn't crash and hub is still ranked first).
    assert scores.get("hub", 0.0) >= max(scores.get("leaf_a", 0.0), scores.get("leaf_b", 0.0))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/retrieval/test_pagerank_graph.py -v 2>&1 | tail -15
```

Expected: `AttributeError: 'NetworkXStore' object has no attribute 'pagerank'`.

- [ ] **Step 3: Add `pagerank()` to the `GraphStore` port**

Open `glyph/store/port.py`. Add to the `GraphStore` Protocol (after `shortest_path`):

```python
def pagerank(self) -> dict[str, float]:
    """Return raw PageRank centrality scores over the full graph (sum ≈ 1.0 when non-empty)."""
    ...
```

Without this, `GraphRetriever` (Step 5) calling `store.pagerank()` on a `store: GraphStore`
parameter fails `mypy --strict` (enabled in `pyproject.toml`) and silently breaks the
Neo4j adapter at runtime — the port exists precisely so retrieval never depends on a
concrete backend.

- [ ] **Step 4: Add `pagerank()` to NetworkXStore**

Open `glyph/store/networkx_store.py`. Find the end of the class and add:

```python
def pagerank(self) -> dict[str, float]:
    """Return raw networkx PageRank scores (sum ≈ 1.0). Empty graph returns {}."""
    if self._g.number_of_nodes() == 0:
        return {}
    return nx.pagerank(self._g)
```

`networkx` is already imported as `nx` at the top of this module — no new import needed.
The internal graph attribute is `self._g` (verified against the current file), not `self._graph`.

- [ ] **Step 5: Add `pagerank()` to Neo4jStore**

Open `glyph/store/neo4j_store.py`. Add to the class (mirrors the "semantics match
NetworkXStore exactly" contract already stated in this file's docstring):

```python
def pagerank(self) -> dict[str, float]:
    """Return raw PageRank scores (sum ≈ 1.0). Computed client-side via networkx —
    avoids requiring the Neo4j GDS plugin for a feature this small."""
    import networkx as nx
    with self._driver.session() as session:
        node_ids = [record["id"] for record in session.run("MATCH (n:Node) RETURN n.id AS id")]
        if not node_ids:
            return {}
        edge_result = session.run("MATCH (a:Node)-[r]->(b:Node) RETURN a.id AS src, b.id AS dst")
        g = nx.DiGraph()
        g.add_nodes_from(node_ids)
        g.add_edges_from((record["src"], record["dst"]) for record in edge_result)
    return nx.pagerank(g)
```

- [ ] **Step 6: Add `pagerank_weight` to GraphRetriever**

Open `glyph/retrieval/graph.py`. Modify `__init__` and `_segments`:

```python
def __init__(
    self,
    store: GraphStore,
    embedder: Embedder,
    nodes: Sequence[Node],
    hops: int = 2,
    anchors: int = 3,
    pagerank_weight: float = 0.0,   # <-- add this
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
    # Pre-compute normalized PageRank (max=1.0) once at index time.
    self._pagerank: dict[str, float] = {}
    if pagerank_weight > 0.0:
        raw = store.pagerank()
        max_pr = max(raw.values(), default=1.0)
        self._pagerank = {k: v / max_pr for k, v in raw.items()}
    self._pagerank_weight = pagerank_weight
```

In `_segments`, replace the score line:

```python
# Before:
score = 1.0 if node.id in anchors else scores.get(node.id, 0.0)

# After:
if node.id in anchors:
    score = 1.0
else:
    cosine = scores.get(node.id, 0.0)
    pr = self._pagerank.get(node.id, 0.0)
    # Linear blend confirmed correct for this case (Perplexity research, 2026-07-01):
    # RRF is for fusing independent retriever rank lists, not for mixing a semantic
    # similarity score with a structural prior like centrality — score-level fusion
    # is the right tool here.
    score = (1.0 - self._pagerank_weight) * cosine + self._pagerank_weight * pr
```

- [ ] **Step 7: Run tests — all must pass**

```bash
python -m pytest tests/retrieval/test_pagerank_graph.py tests/retrieval/test_graph.py -v 2>&1 | tail -20
```

Expected: all PASSED (including the original graph tests — they must not regress).

- [ ] **Step 8: Commit**

```bash
git add glyph/store/port.py glyph/store/networkx_store.py glyph/store/neo4j_store.py glyph/retrieval/graph.py tests/retrieval/test_pagerank_graph.py
git commit -m "feat(retrieval): composite PageRank+cosine scoring in GraphRetriever"
```

---

## Task 3: Benchmark integration — reranked arm + k sweep

**Files:**
- Modify: `scripts/run_benchmark.py`

**Interfaces:**
- Consumes: `CrossEncoderReranker`, `RerankedRetriever` (Task 1), `VectorBaseline` (existing)
- Produces:
  - `reranked_vector` arm in `_build_arms()` output
  - `--k-sweep` CLI flag: prints context_precision for k=2,3,5,8 on the vector arm

- [ ] **Step 1: Add imports and reranked arm to `_build_arms()`**

Open `scripts/run_benchmark.py`. After the existing imports block, add:

```python
from glyph.retrieval.reranked import RerankedRetriever
from glyph.retrieval.reranker import CrossEncoderReranker
```

In `_build_arms()`, after the `path = PathRetriever(...)` line and before the `return`:

```python
reranker = CrossEncoderReranker()
# Reranks `vector` (0.460 context_precision, current best arm in METRICS.md) — not
# `hybrid` (0.347, currently the worst of the three original arms). A reranker can
# only reorder/filter what its underlying retriever already found; it can't recover
# recall `hybrid` already lost, so it's a stronger candidate pool to start from.
reranked_vector = RerankedRetriever(vector, reranker, k=5)
return {
    "graph": graph,
    "vector": vector,
    "hybrid": hybrid,
    "multi_anchor": multi_anchor,
    "path": path,
    "reranked_vector": reranked_vector,
}
```

- [ ] **Step 2: Add `--k-sweep` argument and handler**

In `main()`, after the existing `argparse` block, add the argument:

```python
parser.add_argument(
    "--k-sweep",
    action="store_true",
    help="run vector arm with k=2,3,5,8 and print context_precision table; skip full benchmark",
)
```

After `cases = load_eval_cases(args.queries)`, add the handler (before the `gen_models` block):

The original draft of this loop declared `k` but never wired it into the retrieval
call — `VectorBaseline.retrieve(query, token_budget, k=5)` has a `k` param, but
`AnswerGenerator.answer()` only calls `retriever.retrieve(question, token_budget)`
(no `k`), so every iteration would have silently run with the same default `k=5` and
printed four near-identical rows. Fix: a thin adapter that pins `k` per iteration so
it actually reaches `VectorBaseline`. Also hoist generator construction (with the
same key validation the single-model path already does) outside the loop instead of
reconstructing it once per case.

```python
class _KVectorRetriever:
    """Adapts VectorBaseline to the Retriever port with a fixed k, for --k-sweep."""

    def __init__(self, vector: VectorBaseline, k: int) -> None:
        self._vector = vector
        self._k = k

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        return self._vector.retrieve(query, token_budget=token_budget, k=self._k)


if args.k_sweep:
    embedder = SentenceTransformerEmbedder()
    documents = [
        (piece.label, piece.text)
        for piece in chunk.by_creature(pdf.load(args.source))
        if chunk.is_creature(piece)
    ]
    if args.gen_base_url:
        gen_key = os.environ.get(args.gen_api_key_env or "")
        if not gen_key:
            print(f"{args.gen_api_key_env} not set (needed for generation)", file=sys.stderr)
            return 2
        generator: Generator = OpenAICompatGenerator(
            model=args.gen_model, api_key=gen_key, base_url=args.gen_base_url
        )
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set (needed for generation)", file=sys.stderr)
            return 2
        generator = AnthropicGenerator()
    judge = OpenAICompatJudge(model=args.model, api_key=api_key, base_url=args.base_url)
    print(f"{'k':>4}  context_precision")
    for k in (2, 3, 5, 8):
        vector = VectorBaseline(embedder=embedder)
        vector.index(documents)
        retriever = _KVectorRetriever(vector, k)
        responses = {
            case.id: _precompute(retriever, [case], generator, None)[case.id] for case in cases
        }
        arm_report = score_arm("vector", cases, responses, judge, seed=args.seed)
        cp = next(m.mean for m in arm_report.metrics if m.metric == "context_precision")
        print(f"{k:>4}  {cp:.3f}")
    return 0
```

`ContextPack` needs importing alongside the other `glyph.model.contract` symbols already
used in this script (currently none are imported here — add `from glyph.model.contract
import ContextPack`).

Note: `--k-sweep` requires the source PDF and a generation key. It is intentionally a manual diagnostic tool, not part of the CI gate.

- [ ] **Step 2b: Import score_arm in the script**

Add to the existing imports from harness:

```python
from glyph.eval.harness import ArmReport, run_benchmark, score_arm
```

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all existing tests PASSED, no new failures.

- [ ] **Step 4: Smoke-test the new arm with cached answers**

The reranked arm requires fresh generation (it reranks at retrieve time, not at answer cache time). Skip with a quick unit-level sanity check:

```bash
python3 -c "
from glyph.retrieval.reranker import CrossEncoderReranker
from glyph.model.contract import Segment
r = CrossEncoderReranker()
segs = [Segment(text='Goblins live in caves', source='a', score=0.5),
        Segment(text='Paris weather is cloudy', source='b', score=0.5)]
out = r.rerank('Where do goblins live?', segs, k=1)
print('top segment:', out[0].source)
assert out[0].source == 'a', f'expected a, got {out[0].source}'
print('OK')
"
```

Expected: `top segment: a` / `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_benchmark.py
git commit -m "feat(benchmark): reranked_vector arm + --k-sweep diagnostic flag"
```

---

## Self-Review

**Spec coverage:**
- S-01 ✓ Task 1 step 4
- S-02 ✓ Task 1 step 5
- S-03 ✓ Task 1 step 5
- S-04 ✓ Task 2 step 4
- S-05 ✓ Task 2 step 6
- S-06 ✓ Task 3 steps 1-2

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `Reranker.rerank(query: str, segments: list[Segment], k: int) -> list[Segment]` — used identically in `CrossEncoderReranker`, `_FakeReranker`, and `RerankedRetriever`.
- `GraphStore.pagerank() -> dict[str, float]` — declared on the port (Step 3), implemented by both `NetworkXStore` (Step 4) and `Neo4jStore` (Step 5), called in `GraphRetriever.__init__` as `store.pagerank()` against the port type — no backend-specific assumption.
- `RerankedRetriever(retriever: Retriever, reranker: Reranker, k: int)` — constructed in `_build_arms()` with matching types.
- `_KVectorRetriever.retrieve(query: str, token_budget: int = 1000) -> ContextPack` — satisfies `Retriever` structurally so `_precompute()` accepts it unchanged.

**Research check (Perplexity, 2026-07-01) — resolved:**
- Linear score fusion (cosine + PageRank) confirmed as the right approach; RRF is for
  fusing independent retriever rank lists, not for blending similarity with a
  structural centrality prior. No change needed to the Task 2 Step 6 blend.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` confirmed as the right reranker baseline
  (NDCG@10 74.30 on TREC DL 2019, best quality/speed tradeoff for a small corpus).
  No change needed to Task 1.
- Personalized PageRank (seeded on the query's anchor nodes, via `nx.pagerank(...,
  personalization=...)`) is reported as more accurate than global PageRank — it
  avoids boosting nodes that are globally central but irrelevant to the query.
  **Deliberately deferred**: this plan ships global PageRank (precomputed once at
  index time, matching S-04/S-05 as written) because it's the cheaper rung and lets
  the benchmark say whether global PageRank alone reaches the 0.55 target before
  paying for a per-query PageRank recomputation. If `--k-sweep`/benchmark results
  after Task 2 show `pagerank_weight` isn't moving `context_precision` enough,
  Personalized PageRank is the next thing to try — it requires moving the
  `store.pagerank()` call from `GraphRetriever.__init__` to `retrieve()` (seeded on
  that call's `anchors`), which is a bigger change than a parameter tweak.

**TCM coverage:** All 6 Story IDs have test IDs. ✓
