# P1.4 — Cost-Gate Results (Monster Manual)

**Date:** 2026-06-10
**Source:** Monster Manual (PT-BR), 351 pages
**Model:** Claude Haiku 4.5 (`claude-haiku-4-5`), structured extraction via `messages.parse`
**Filter:** `chunk.is_creature` — only chunks with attribute block (FOR/DES/CON/INT/SAB/CAR);
front-matter, rules, and index discarded (767 chunks → 425 creatures).

## Measured Metrics

| Metric | Value |
|---|---|
| Chunks (creatures) | 425 |
| Nodes | 693 (458 `ENTITY` / 235 `CONCEPT`) |
| Edges | 1.305 |
| Input tokens | 739.649 |
| Output tokens | 94.258 |
| **Cost** | **US$ 1,2109** ($1/M input + $5/M output) |
| Latency | 1.048,4 s (~17,5 min), 2,47 s/chunk |

Edge distribution: `immune_to` 768, `resists` 395, `vulnerable_to` 47, `summons` 49,
`inhabits` 46. All five documentary relation types from ADR-G2 are exercised.

Cost came **below estimate** (~US$1.50): structured output is compact (94K tokens
output, not the ~255K estimated).

## Quality (Manual Sampling)

Relations mostly correct:
- Deva/Planetar/Solar (angels): `resists radiant/bludgeoning/piercing/slashing`,
  `immune_to charmed` — correct.
- Apparition/Specter (undead): `resists acid/lightning/fire/cold/thunder` — correct.
- Aarakocra: no relations — correct.

Probabilistic error observed (expected, this is what the benchmark measures):
- ANKHEG: `resists acid` — likely confusion between acid *attack* and acid *resistance*.

Cosmetic limitation: node labels come with inconsistent casing (`ANKHEG`, `abolete`,
`Deva`) because the LLM returns names in varying cases. IDs are normalized (lowercase),
so deduplication works; only labels lack uniformity. Candidate for label normalization
in a next iteration.

## Conclusion

Gate approved: US$ 1.21 for the complete bestiary, well within budget; graph dense in
relations and with sufficient quality for Phase 2 (retrieval + vector baseline). **Do not
scale** to PHB/DMG in this phase (P1.5) — those books require their own schema.

Reproduction: `python3 scripts/extract_book.py "<Monster Manual.pdf>" out/monster-manual.json`
with `ANTHROPIC_API_KEY` set. Graph persisted in `out/monster-manual.json`.
