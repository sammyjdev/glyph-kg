# KICKOFF — Phase 0: Foundation

Handoff document for agent-assisted execution. Defines context, contracts, and done definition per sub-task. Agent executes under architectural direction; does not invent scope outside this document.

## Minimum context

GLYPH is a knowledge graph library for documents and code (see SPEC.md, ARCHITECTURE.md). Phase 0 delivers the skeleton: structure, domain model, ports, and quality gates. No extraction, no retrieval yet. Goal: installable lib with contracts in place and CI green.

Read before starting: `SPEC.md`, `docs/ARCHITECTURE.md`, `docs/decisions/dec-g1-extractor-port-and-backend.md`.

## Stack

- Python 3.11+
- Pydantic v2 (domain model)
- NetworkX (default store adapter)
- pytest + pytest-cov (tests)
- ruff + mypy (lint, types)
- tree-sitter comes in Phase 4; not in this phase

## Target file structure

```
glyph-kg/
  pyproject.toml
  glyph/
    __init__.py
    model/
      __init__.py
      node.py          # Node, NodeType
      edge.py          # Edge, EdgeType
    store/
      __init__.py
      port.py          # GraphStore Protocol
      networkx_store.py
    extract/
      __init__.py
      port.py          # Extractor Protocol
  tests/
    architecture/
      test_dependencies.py   # import invariants
    model/
    store/
```

## Sub-tasks

### P0.1 — Scaffold
- Create `pyproject.toml` with pip-installable build, deps (pydantic, networkx) and dev-deps (pytest, pytest-cov, ruff, mypy).
- Package structure above with `__init__.py`.
- **Done:** `pip install -e ".[dev]"` works; `import glyph` works.

### P0.2 — Domain model
- `Node` (id, type, label, attrs), `NodeType` (enum covering code and document), `Edge` (src, dst, type, attrs), `EdgeType` (enum with code and document types separated).
- Pydantic v2, immutable where it makes sense (frozen).
- **Done:** models validate; round-trip serialization tests pass; `model` imports nothing outside pydantic and stdlib.

### P0.3 — GraphStore port
- `Protocol` with `upsert_nodes`, `upsert_edges`, `neighbors(node, hops)`, `subgraph(seed, hops)`, `shortest_path(src, dst)`.
- Return types defined (`Subgraph`, `Path`).
- **Done:** Protocol imports only from `model`; mypy validates.

### P0.4 — NetworkX adapter
- Implements `GraphStore` over an `nx.DiGraph`.
- Persistence: `save(path)` / `load(path)`.
- **Done:** store contract suite passes (upsert, neighbors with hops, subgraph, shortest_path, persistence round-trip).

### P0.5 — Extractor port
- `Protocol` with `extract(source) -> tuple[Sequence[Node], Sequence[Edge]]`.
- Define `Source` type (union of path/string as needed).
- **Done:** Protocol imports only from `model`; no concrete adapter yet (coming in Phases 1 and 4).

### P0.6 — Quality gates
- CI (GitHub Actions): lint (ruff), types (mypy), test (pytest), coverage gate.
- `tests/architecture/test_dependencies.py`: verifies ARCHITECTURE.md invariants (model does not import extract/store/retrieval; adapters do not import each other).
- **Done:** CI green on first PR; coverage gate active; invariant test fails if anyone violates the import rule.

## Phase 0 done definition

- `pip install -e .` works; `import glyph` exposes model, store port, extract port.
- NetworkX store passes contract suite.
- CI green with lint, types, test, coverage, and architecture invariants.
- ADR-G1 committed (already in place).
- No extraction or retrieval logic yet; that is Phase 1+.

## Out of scope in this phase

DocumentExtractor, CodeExtractor, retrieval, vector baseline, GNOMON, AXON integration. Do not anticipate. Each has its phase in GLYPH_PLAN.md.

## Conventions

- TDD: test before implementation in each sub-task.
- Small commits per sub-task, imperative message style.
- New architectural decision during execution: stop and open ADR, do not decide inline.
