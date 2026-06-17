# GLYPH — Execution Plan

> **Status (2026-06-11): Phases 0–7 complete and in `main`.** Benchmark run across both domains
> (document: graph leads faithfulness at lower cost; code: vector beats graph) — numbers in
> [METRICS.md](../METRICS.md) and [METRICS-code.md](../METRICS-code.md). Open/future (non-blocking):
> P1.5 (scale document corpus), quality backlog (real tokenizer, casing), and follow-ups from P7
> (real summarization round + global benchmark arm). The plan below is the original record.

> Unified knowledge graph library: common graph core, two pluggable extractors (document via LLM, code via tree-sitter), graph-aware retrieval, measured against vector baseline with GNOMON. Document-first (covers job requirements), code later (on-brand AXON).

## Target architecture

```
                      Extractor port
              +-------------+-------------+
              |                           |
     DocumentExtractor             CodeExtractor
     (LLM, probabilistic)          (tree-sitter, deterministic)
              |                           |
              +----------> Graph <--------+
                          (Node/Edge model)
                              |
                       GraphStore port
                    +---------+---------+
                    |                   |
              NetworkX (default)   Neo4j (adapter)
                              |
                    graph-aware retrieval
                              |
              GNOMON benchmark  vs  vector baseline (Python, same corpus)
```

Invariants (ArchUnit-style):
- Extractors implement the port; graph core does not know concrete extractor.
- Retrieval and store depend only on domain model, not on extractor.
- Vector baseline is fair and real implementation, not a strawman.

## Closed decisions

- Strong version: common core + Extractor port with two adapters. Not a single schema serving both domains.
- Backend: GraphStore port, NetworkX default + Neo4j adapter (implemented, smoke-tested — 14/14 contract tests vs Neo4j 5).
- Order: DocumentExtractor first (driver = cover job requirements), CodeExtractor later.
- Document extraction: LLM-based (justified for prose).
- Consumed by AXON via `GraphContextSource` (ADR-102), which delegates to GLYPH.

## Dependencies and blockers

- **GNOMON (audited): already pip-installable** (hatchling, src-layout; `run_eval` and `aggregate_metric` importable, not locked in `__main__`). **No longer blocks Phase 3.** The only item is GLYPH referencing it by path/git in `pyproject` (no PyPI/remote) — one line, sub-task of Phase 3, not its own phase.
- **Corpus:** 10–15 D&D books (150–300 pages). Real data, dense in entities. LLM extraction will be expensive at scale; measure cost on 1 book before scaling (gate in Phase 1).
- **Volatile APIs** (LLM extraction, NetworkX, Neo4j driver, PDF lib): confirm in current docs before each implementation phase.

---

## Phase 0 — Foundation

Goal: library skeleton, domain model, ports, quality gates. No extraction yet.

- **P0.1** Repo scaffold: `pyproject.toml` (pip-installable), hexagonal structure (`glyph/model`, `glyph/store`, `glyph/extract`, `glyph/retrieval`, `glyph/eval`), license, README stub.
- **P0.2** Domain model (Pydantic v2): `Node`, `Edge`, `EdgeType`, `NodeType`. Generic enough for code and document, specific enough not to be soup. Edge types separated by domain (`CALLS`/`IMPORTS` for code; `RELATES_TO`/`MENTIONS`/etc. for document).
- **P0.3** `GraphStore` port (Protocol): `upsert_nodes`, `upsert_edges`, `neighbors(node, hops)`, `subgraph(seed, hops)`, `shortest_path`.
- **P0.4** NetworkX adapter (default): implements port in-memory with persistence (pickle/graphml).
- **P0.5** `Extractor` port (Protocol): `extract(source) -> tuple[Sequence[Node], Sequence[Edge]]`.
- **P0.6** Quality gates: pytest + coverage gate, architecture invariant test (core does not import concrete extractor), CI (lint, types, test).
- **ADR-G1**: Extractor port + backend choice (NetworkX default, Neo4j adapter, pip-installable rationale).

Deliverable: empty installable lib with model, ports, and NetworkX. Content: none yet.

---

## Phase 1 — Document extraction (priority, covers job requirements)

Goal: build real document KG from D&D corpus.

- **P1.1** PDF ingestion in Python (pymupdf or similar): PDF → text by section/page, with metadata (book, page).
- **P1.2** Document chunking (structure-aware: chapter/section, not blind cut).
- **P1.3** `DocumentExtractor` (LLM): entity/relation extraction prompt, parse to `Node`/`Edge`. D&D domain entity schema (creature, spell, item, rule, location) and relations (resists, requires, belongs to, etc.).
- **P1.4** Cost/quality gate on 1 book: run P1.1–1.3 on single book, measure extraction cost (tokens × price), latency, and manual sampling of relation quality. **Do not scale before approving this gate.**
- **P1.5** Scale to full corpus (10–15 books), persist graph via NetworkX.
- **ADR-G2**: document extraction strategy (entity schema, prompt model, declared probabilistic limitation).

Deliverable: document KG persisted on real corpus. Content: short post on document KG extraction.

---

## Phase 2 — Retrieval + fair vector baseline

Goal: graph-aware retrieval and the baseline against which to measure. Both sides of the comparison.

- **P2.1** Graph-aware retrieval: given a query, anchor entities and expand neighborhood by `hops`, return structural context.
- **P2.2** Vector baseline (Python, same corpus): chunk + embedding + vector store (in-memory or pgvector). Real and fair implementation, equal care as graph path. This is the experiment control.
- **P2.3** Hybrid mode: graph + vector fusion (third benchmark arm).
- **P2.4** Unified output contract (`Segment`/`ContextPack`) for three modes, comparable token-to-token.
- **ADR-G3**: baseline design (why it is fair, parameters, budget equality across arms).

Deliverable: three functional retrieval modes on same corpus. Content: none until measured.

---

## Phase 2.5 — Unblock GNOMON (RESOLVED by audit)

The GNOMON audit dissolved this phase: GNOMON **already** is pip-installable and `run_eval`/
`aggregate_metric` are truly importable. No packaging to do (decision A vs B falls away).
What remains is small and migrates to Phase 3 as P3.0: reference GNOMON by path/git in
GLYPH `pyproject` and write the `RagTarget` adapter (see below). This phase no longer blocks.

---

## Phase 3 — Benchmark + reproducible baseline

Goal: the honest number. This phase is the job requirement claim and the article.

- **P3.0** Consume GNOMON: reference by path/git in GLYPH `pyproject`; write a **`RagTarget`** adapter
  (GNOMON is pull-based — `run_eval` calls `target.query(question)`). GLYPH
  pre-computes results for each arm (graph, vector, hybrid) keyed by question and exposes a
  `RagTarget` per arm that returns the stored result via `query(question) -> RagResponse`.
  Runs `run_eval` once per arm. (Push-based `evaluate()` in GNOMON is future improvement — GNOMON
  ROADMAP, not GLYPH.) **GNOMON v1 contract restrictions, declared:**
  - `RagResponse` requires `total_tokens` and `latency_ms` (validated ≥ 0) → **instrumenting real token and
    latency across three arms is required**, not optional (placeholder pollutes the report).
  - v1 metrics = **`faithfulness` and `context_precision`** only (answer relevance and context recall
    are v2, not built). Article reports only these two, with CIs — do not promise recall.
  - **Currency cost does not exist in GNOMON**, only tokens; GLYPH calculates USD from `total_tokens`
    separately.
- **P3.1** Benchmark harness: runs three modes (graph, vector, hybrid) over a set of queries from D&D corpus.
- **P3.2** Query set: realistic query set (questions requiring relations between entities, where graph should win; and simple factual, where vector may suffice). Report both.
- **P3.3** Metrics via GNOMON: `faithfulness` and `context_precision` (the two from v1), plus token efficiency, cost (calculated by GLYPH from tokens) and latency. All with CI (percentile bootstrap).
- **P3.4** Reproducible baseline (lessons C1b+3c): fixed versioned dataset + `make benchmark` that regenerates aggregates; reference number in `METRICS.md` with regression check (build fails if diverges beyond tolerance). Corpus/queries in frozen state for reproducibility.
- **P3.5** Honest report: table with CIs including where graph **did not** win; declare n (queries), CI width, and whether tokens are real count or estimate.
- **Extraction quality backlog** (observed at gate P1.4): normalize label casing and evaluate probabilistic relation errors (e.g., ANKHEG `resists acid`). Detail in [decisions/phase3-quality-backlog.md](decisions/phase3-quality-backlog.md).
- **ADR-G4**: eval methodology (query set, bootstrap, relevance definition).

Deliverable: reproducible results table. Content: main article ("GraphRAG vs vector retrieval on real corpus, measured").

---

## Phase 4 — Code extraction (on-brand, AXON)

Goal: second extractor, second domain, second KG claim.

- **P4.1** `CodeExtractor` (tree-sitter): reuse/align with existing AXON logic (`graph_extractor.py`) rather than reimplement from scratch. Decision: GLYPH becomes canonical source and AXON delegates, or GLYPH mirrors behavior. Resolve in P4.1.
- **P4.2** Target languages: Python + Java (AXON's indexed set), TypeScript optional.
- **P4.3** Validate on real repos at fixed SHA (AXON, GNOMON), generate code graph.
- **P4.4** Run same benchmark harness (Phase 3) on code domain: structural context vs vector for agent tasks.
- **ADR-G5**: symbol resolution in code extractor (limitation by name/import graph declared).

Deliverable: validated code KG. Content: post comparing two domains in same framework.

---

## Phase 5 — AXON integration

Goal: close the loop with already-specified rescope (ADR-102/103).

- **P5.1** AXON `GraphContextSource` (ADR-102) delegates to GLYPH lib as dependency.
- **P5.2** Update ADR-102/103: one-line note stating `GraphContextSource` is implemented by GLYPH.
- **P5.3** AXON graph consolidation (ADR-103) remains; GLYPH consumes consolidated SQLite graph.
- **P5.4** Integration test: AXON serves graph-aware context via GLYPH over MCP.

Deliverable: AXON consuming GLYPH. Content: post on context-source integration.

---

## Phase 6 — Publication / portfolio

Goal: convert work into verifiable evidence.

- **P6.1** GLYPH README with verified metrics, declared limitations, reproducible benchmark link.
- **P6.2** Technical article (validation-first: claim checklist against codebase, number verified before publishing).
- **P6.3** CV/LinkedIn claim: "built a knowledge-graph library spanning document and code domains; benchmarked graph-aware vs vector retrieval with confidence intervals." Covers Near and Marlabs to the letter.
- **P6.4** Endorsement/visibility: showcase in Discords (Qdrant, MCP), link in first comment, not in post body.

Deliverable: publishable portfolio. Covers KG gap in job requirements with measured evidence.

---

## Critical sequence

```
Phase 0  ->  Phase 1  ->  Phase 2  ->  Phase 3 (job requirement claim)
                                          |    (GNOMON ready; P3.0 = reference + RagTarget adapter)
                                          |
                    Phase 4 (code)  -------+
                          |
                    Phase 5 (AXON)  ->  Phase 6 (publication)
```

The path to Phase 3 is what covers the job requirements. Phases 4–5 strengthen the moat and do not block the document claim. Each phase has an isolated publishable deliverable: you accumulate content without waiting for the end.

## Global gates

- TDD per sub-task; nothing enters without test.
- Architecture invariant verified in CI.
- Each phase closes with its ADR before advancing.
- Cost measured early (P1.4) before scaling extraction.
- Fair baseline is a validity condition for the benchmark, not optional.
- Published number is reproducible from repo (P3.4), otherwise not published.

## Open before Phase 0

- Name: unified GLYPH confirmed. Keep.
- ADR numbering: ADR-G1..G5 are internal to GLYPH (own repo); independent from AXON's dec-NNN.
- Embedding lib and vector store for baseline (P2.2): define in Phase 2 (options: sentence-transformers + pgvector, or what GNOMON/AXON already use, for consistency).
