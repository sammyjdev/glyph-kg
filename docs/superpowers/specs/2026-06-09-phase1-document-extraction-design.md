# Design — Phase 1: Document Extraction (P1.1–P1.4)

**Date:** 2026-06-09
**Status:** Approved (brainstorming)
**Scope of this session:** P1.1 → P1.4. P1.5 (scaling the corpus) is deferred to a future session.

## Goal

Build a real document-based knowledge graph from a Dungeons & Dragons book under the existing
`Extractor` port and measure the LLM extraction cost on one book before scaling. Deliverables:
Monster Manual KG persisted via NetworkX with real cost and latency numbers, ADR-G2 committed,
TDD suite green, and CI passing.

## Corpus

The three Portuguese D&D 5e core books are available locally and are the target corpus:

| Book | Path | Role in Phase 1 |
|---|---|---|
| Monster Manual | `~/Downloads/3-Monster Manual.pdf` | **Anchor.** Cost gate (P1.4) runs on this. |
| Player's Handbook | `~/Downloads/1-Player's Handbook.pdf` | Registered; enters future phase with its own schema. |
| Dungeon Master Guide | `~/Downloads/2-Dungeon Master Guide.pdf` | Registered; enters future phase with its own schema. |

These paths are not committed to the repo (PDFs are gitignored and are not ours to distribute).
The benchmark harness references the path via environment variable or argument.

### PDF Reality (probed before design)

- Monster Manual: 351 pages, **extractable text** (not a scan), ~500–700 words/page.
- Language: **Portuguese** ("Manual dos Monstros"). Schema and prompt in Portuguese; enums in English.
- **No TOC** in the PDF. Creatures appear as **ALL-CAPS headings** at the start of each entry
  ("ABOCANHADOR MATRAQUEANTE", "DUERGAR", "KUO-TOA"), with page number.

## Closed Decisions

1. **LLM Provider:** Claude Haiku 4.5 (`claude-haiku-4-5`), $1/$5 per 1M tokens (input/output),
   200K context. README default (`GLYPH_LLM_PROVIDER=anthropic`). Provides real cost numbers
   for the benchmark.
2. **Ingestion:** local text via pymupdf (P1.1), not PDF-native/vision. Cheap, deterministic,
   full chunk control — what GLYPH_PLAN P1.1 describes.
3. **Chunking:** by entry via header detection (ALL-CAPS / font), with fallback to page-window
   where detection has low confidence. Each creature ≈ one chunk ≈ one LLM call.
4. **Schema:** MM-focused. Creature = `NodeType.ENTITY`; damage type / condition / habitat =
   `NodeType.CONCEPT`. Relations exercised by MM (see ADR-G2).
5. **Structured output:** `messages.parse` with Pydantic schema (Haiku supports structured
   outputs), mapped to `Node`/`Edge`.

## Architectural Fit

Phase 0 hexagonal architecture preserved. New package `glyph/extract/document/` — an adapter
of the `Extractor` port. Imports only from `glyph.model`, `glyph.extract.port`, and external
libs (pymupdf, anthropic). **Does not import `glyph.store`** — the Phase 0 invariants test
(`tests/architecture/test_dependencies.py`) enforces the rule.

```
glyph/extract/document/
  __init__.py
  pdf.py          # P1.1: PDF -> [Page(book, number, text, spans)]
  chunk.py        # P1.2: [Page] -> [Chunk(label, text, book, pages)] by entry
  schema.py       # Pydantic: ExtractedEntity / ExtractedRelation (LLM output contract)
  prompt.py       # extraction system prompt (Portuguese) + few-shot
  llm.py          # thin Anthropic adapter (Haiku, structured output, usage capture)
  extractor.py    # DocumentExtractor: implements Extractor; orchestrates pdf->chunk->LLM->Node/Edge
  cost.py         # P1.4: tokens × price, latency, aggregation
```

## Model — EdgeType Extension (ADR-G2)

`EdgeType` (document domain) gains the types exercised by MM, in addition to existing ones
(`RESISTS`, `RELATES_TO`, `MENTIONS`, `REQUIRES`):

- `IMMUNE_TO` — immune to (damage/condition)
- `VULNERABLE_TO` — vulnerable to (damage)
- `INHABITS` — inhabits (location/plane)
- `SUMMONS` — summons (another creature)

`NodeType` **does not change**. Graph schema:

```
ENTITY(creature) --RESISTS/IMMUNE_TO/VULNERABLE_TO--> CONCEPT(fire, cold, poison, stunned, ...)
ENTITY(creature) --INHABITS--> CONCEPT(Underground, plane, ...)
ENTITY(creature) --SUMMONS--> ENTITY(summoned creature)
```

ADR-G2 records the schema, the justification for the MM-focused scope, and the **declared
probabilistic limitation**: document extraction has error; that is why the benchmark measures
quality rather than assuming correctness. ADR committed before implementation (CONTRIBUTING rule).

## Data Flow

`build(source=MM.pdf)`:
1. `pdf.load(source)` → pages with text and spans (font/size per line) + metadata (book, page).
2. `chunk.by_creature(pages)` → chunks per entry (ALL-CAPS header/font; fallback page-window).
3. For each chunk: `llm.extract(chunk)` → structured response (Haiku, `messages.parse`,
   **system prompt cached** across chunks to reduce cost), capturing `usage`.
4. `map(extracted)` → `Node`/`Edge`, with **dedup** (same creature/concept = one node per normalized id).
5. Persist via `NetworkXStore.save(path)`.

## LLM Output Contract (schema.py)

Pydantic, aligned with structured output constraints (no recursion, `additionalProperties:false`):

```
ExtractedEntity:  name: str   kind: "creature" | "concept"   attrs: dict (CR, type, alignment ...)
ExtractedRelation: subject: str   predicate: RESISTS|IMMUNE_TO|VULNERABLE_TO|INHABITS|SUMMONS   object: str
ExtractionResult:  entities: list[ExtractedEntity]   relations: list[ExtractedRelation]
```

The mapping `ExtractionResult -> (Node[], Edge[])` is pure logic, testable without network.

## Test Strategy (TDD)

Mock only at the API boundary (non-deterministic and paid — justified exception to TDD skill).
All domain logic is tested with real code.

- **pdf.py** — minimal PDF fixture generated in test (pymupdf) → asserts pages, text, metadata.
- **chunk.py** — synthetic text fixtures with headers → asserts entry boundaries and fallback.
- **schema.py / mapping** — canonical `ExtractionResult` → expected `Node`/`Edge` + dedup. Pure.
- **extractor.py** — injects a *fake LLM* (returns fixed `ExtractionResult`) → tests orchestration
  pdf→chunk→map without network.
- **cost.py** — pure function (tokens, price) → cost/latency.
- **llm.py (real adapter)** — one live smoke test marked `@pytest.mark.live`, skipped by default
  (runs only with `ANTHROPIC_API_KEY`). The P1.4 gate run is the real live validation.

CI stays green: `live` tests excluded from default; coverage/lint/types/invariants maintained.

## P1.4 — Cost Gate (Paid Run)

Runs the pipeline on the entire Monster Manual and measures:
- **Cost:** Σ(input_tokens × $1/1M + output_tokens × $5/1M), read from `response.usage`.
- **Latency:** total and per-chunk time.
- **Volume:** number of extracted entities and relations.
- **Quality:** sample ~10 creatures for manual verification of relations.

Persists the resulting graph. **Estimate: ~US$1–3** (≈290K tokens of text + prompt overhead;
structured output per creature). **Run preconditions:** `ANTHROPIC_API_KEY` exported (not
currently) and explicit user approval before paid call. Discipline gate: **do not scale** to
PHB/DMG in this session (that is P1.5).

## Dependencies and Packaging

Optional extra `glyph-kg[document]` = `pymupdf`, `anthropic` — keeps base lib lightweight
(DX value of ADR-G1). Env vars: `ANTHROPIC_API_KEY` (required for extraction/gate),
`GLYPH_LLM_PROVIDER=anthropic` (already in README).

## Phase 1 Done Criteria (this session)

- ADR-G2 committed before implementation.
- `EdgeType` extended; model validates; round-trip maintained.
- `glyph/extract/document/` implemented with TDD; suite green; architecture invariants intact.
- Live smoke test passes with key (executed once).
- P1.4 gate run on MM with user approval: cost/latency/volume reported, ~10 creatures sampled,
  graph persisted.
- CI green (lint, types, test, coverage, invariants).

## Out of Scope

P1.5 (scaling to full corpus), PHB/DMG, retrieval (Phase 2), vector baseline, GNOMON, AXON
integration. Broad schema (magic/item/rule/location) deferred to when PHB/DMG enter.
