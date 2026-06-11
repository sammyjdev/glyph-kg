# GLYPH benchmark dataset (Phase 3)

Frozen, versioned inputs for the retrieval benchmark (P3.2 / P3.4). Reproducibility
depends on these files staying fixed between runs.

## `queries.json`

25 hand-authored questions over the Monster Manual graph (`out/monster-manual.json`),
balanced across categories that should and should not favor the graph arm:

| Category | n | Hypothesis |
|---|---|---|
| `relational_single` | 9 | graph favored — one relation hop, precision matters |
| `relational_multi` | 3 | graph strongly favored — intersections across relations |
| `entity_relation` | 6 | graph favored — relation lookup on a single entity |
| `factual_attribute` | 4 | vector favored — attrs live in `node.attrs`, absent from graph segments |
| `factual_description` | 3 | vector favored — open prose, raw chunk text wins |

Each query carries:

- `relevant_sources` / `relevant_labels` — a **candidate** relevance oracle for
  `context_precision`, joined from the KG by `scripts/build_query_set.py`.
- `answer_key` — human-readable expected answer (relation targets or an attribute value);
  `null` for relational queries (the relevant labels are the answer) and open descriptions.
- `graph_favored` — the prior hypothesis, reported honestly in P3.5 whether or not it holds.

### Oracle caveat (honesty, P3.5)

`relevant_sources` is **KG-derived, not verified gold**. It inherits the extraction's
known probabilistic errors — e.g. `ankheg → resists ácido` is likely wrong
(see [`docs/decisions/phase1-cost-gate-results.md`](../docs/decisions/phase1-cost-gate-results.md)).
The `ent-ankheg-resist` query keeps that error on purpose so the benchmark exposes where
the graph carries extraction noise. Verify against the source text before publishing numbers.

## Regenerating

```bash
python3 scripts/build_query_set.py          # rewrite eval/queries.json
python3 scripts/build_query_set.py --check   # CI: fail if the committed file is stale
```

The questions are authored in the generator; only the KG-joined fields are computed.
