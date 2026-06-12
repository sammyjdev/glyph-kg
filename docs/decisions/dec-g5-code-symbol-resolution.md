# ADR-G5: Code symbol resolution in the code extractor

**Data:** 2026-06-11
**Status:** Accepted

## Context

Phase 4 adds the second extractor: code, via tree-sitter (deterministic), behind the same
`Extractor` port as the document extractor. AST parsing is exact; what is heuristic is **linking a usage to
the symbol it references** (call → function, `extends` → class) across files.

## Decision

**Languages (P4.2):** Python and Java (the indexed set from AXON). TypeScript remains a future extension — the
`Grammar` is pluggable (node-types + language-specific name extractors), so adding one is
localized.

**Nodes and edges.** `FILE` (id = relative POSIX path), `CLASS`/`FUNCTION` (id qualified by scope,
e.g. `glyph/eval/cost.py::ArmResponse.total`), `MODULE` (import target). Edges: `DEFINES`
(scope → symbol), `IMPORTS` (file → module), `CALLS` (function → function), `INHERITS` (class →
class). All deduplicated.

**Resolution by unqualified, unique name (the declared limitation).** The extractor builds an index
`simple name → ids defined in corpus` and only emits `CALLS`/`INHERITS` when the target name is
defined **exactly once**. Consequences:
- Ambiguous name (defined in 2+ places) → edge **omitted** (under-approximation, we don't guess the
  most likely one).
- Name not defined in corpus (stdlib, external lib, inherited method from outside) → edge omitted.
- Calls via subscript/expression (`fns[0]()`) → no simple name → omitted.
- No type inference: `obj.method()` resolves by **method name** (`method`), not by type of
  `obj`. In corpora with a unique method name this is correct; with collision, omitted.

**GLYPH as canonical source (P4.1).** AXON is outside the scope of this repo; GLYPH implements the code
extractor standalone and becomes the canonical source for code-graph. When AXON integrates
(Phase 5), it delegates here instead of reimplementing — aligned with AXON's ADR-102/103.

**Code benchmark (P4.4).** The Phase 3 harness, the retrievers (graph/vector), and the
`ContextPack` contract are **domain-agnostic**: they operate on any graph + any chunk texts.
Running the same benchmark on the code domain means pointing the harness at a code-graph + a code
query set — no new evaluation code is needed, only the corpus and queries.

## Consequences

Deterministic and reproducible (same SHA → same graph). High precision, recall limited by design in
cross-symbol resolution — declared, not assumed, and the benchmark measures where this matters. Future
improvement paths (not in this phase): resolution by import graph + scope, type-qualified resolution.
