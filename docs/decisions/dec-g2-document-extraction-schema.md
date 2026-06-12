# ADR-G2: Document extraction schema (Monster Manual)

**Date:** 2026-06-09
**Status:** Accepted

## Context

Phase 1 builds the first documentary knowledge graph for GLYPH from the
Monster Manual (PT-BR, 351 pages, extractable text, no TOC). Extraction is
probabilistic: an LLM reads Portuguese prose and infers entities and relations, with
error. The `EdgeType` from Phase 0 covers only `{RELATES_TO, MENTIONS, REQUIRES,
RESISTS}` for documents — insufficient for the relations that the Monster Manual
actually exercises.

## Decision

**MM-focused schema.** Entity = creature (`NodeType.ENTITY`); damage type,
condition, and location/plane = concept (`NodeType.CONCEPT`). `EdgeType` gains four
document-domain members:

- `IMMUNE_TO` — immunity to damage/condition
- `VULNERABLE_TO` — vulnerability to damage
- `INHABITS` — inhabits location/plane
- `SUMMONS` — summons/conjures another creature

alongside existing `RESISTS`, `RELATES_TO`, `MENTIONS`, `REQUIRES`.
`NodeType` does not change.

Graph schema:

```
ENTITY(creature) --RESISTS/IMMUNE_TO/VULNERABLE_TO--> CONCEPT(fire, cold, stunned, ...)
ENTITY(creature) --INHABITS--> CONCEPT(Underdark, plane, ...)
ENTITY(creature) --SUMMONS--> ENTITY(summoned creature)
```

## Consequences

**Positive:** schema aligns with MM structure, strong signal for the
graph-vs-vector benchmark. Adding a new domain (PHB/DMG) is extending the enum and the prompt,
without touching the core.

**Trade-offs / to observe:** documentary extraction has error — this is why the
benchmark measures quality instead of assuming. The schema does not cover spells/items/rules
(PHB/DMG content); this is declared and deferred to a future phase.

## Alternatives considered

| Alternative | Why it was discarded |
|---|---|
| Reuse only existing EdgeType | Merges immune/resist/vulnerable into a single edge; loses fidelity |
| Broad schema (spells/items/rules/location) now | MM does not exercise these types; extraction noisy, more costly, no gain now |
| Free attribute dict in LLM output | Structured outputs require `additionalProperties:false`; fixed optional fields instead |
