# GraphRAG vs vector retrieval over a real corpus, measured

> **Validation-first draft (P6.2).** Every claim below points at the code or artifact that
> backs it. Claims tagged **[pending run]** need the numbers from a real benchmark
> (`make benchmark` with `GROQ_API_KEY` + `ANTHROPIC_API_KEY`) before publishing — they are
> written as placeholders on purpose, not asserted.

## Thesis

Vector similarity ignores structure: who cites whom, what relates to what. For corpora dense
in entities and relations, a knowledge graph serves context that embeddings miss. GLYPH builds
that graph from two domains — documents (LLM extraction) and code (tree-sitter) — under one
extractor port, and **measures** when the graph beats the vector baseline and when it does not.

## What is actually built (verifiable today)

| Claim | Evidence in repo |
|---|---|
| One graph core, two extractors behind one port | `glyph/extract/port.py`, `glyph/extract/document/`, `glyph/extract/code/` |
| Document KG from a real corpus (Monster Manual) | `out/monster-manual.json` (693 nodes, 1305 edges); cost gate in `docs/decisions/phase1-cost-gate-results.md` ($1.21 measured) |
| Code KG, deterministic, Python + Java | `glyph/extract/code/`; self-hosted `out/glyph-code.json` (236 nodes, 369 edges) |
| Graph-aware retrieval + a fair vector baseline + hybrid, one output contract | `glyph/retrieval/`, `glyph/baseline/vector.py`, `glyph/model/contract.py`; ADR-G3 |
| Benchmark against GNOMON with bootstrap CIs | `glyph/eval/` (target, judge, harness, report); ADR-G4 |
| Frozen, reproducible query set | `eval/queries.json` (n=25), `scripts/build_query_set.py --check` |
| Reproducible baseline + regression gate | `make benchmark` / `make benchmark-check`, `eval/benchmark-baseline.json` |
| AXON consumes GLYPH as the graph source | `glyph/integration/GraphContextSource`; `docs/axon-integration.md` |

## Method (ADR-G4)

Three arms (graph, vector, hybrid) answer the same 25 questions over the same corpus, under the
same token budget and embedder. Each arm generates a grounded answer over its retrieved context
(real token + latency instrumentation). An OpenAI-compatible OSS judge (Groq) scores GNOMON's two
v1 metrics — `faithfulness` and `context_precision` — and we aggregate per-case scores with a
seeded percentile bootstrap. Cost is generation tokens at Haiku 4.5 rates.

## Results **[pending run]**

The table below is generated into `METRICS.md` by the benchmark. Until a real run fills it, no
number is claimed. The honest report will include **every** arm's metric with its CI — explicitly
including the queries where the graph did **not** win (the factual/attribute category, where the
vector baseline is expected to match or beat the graph), plus token efficiency, USD cost and latency.

| Metric | graph | vector | hybrid |
|---|---|---|---|
| faithfulness | _pending_ | _pending_ | _pending_ |
| context_precision | _pending_ | _pending_ | _pending_ |
| total tokens / cost / latency | _pending_ | _pending_ | _pending_ |

## Declared limitations

- **Relevance oracle is KG-derived, not gold.** It inherits extraction errors (e.g.
  `ankheg → resists ácido`), kept on purpose so the comparison exposes graph noise.
- **Two metrics only.** Answer-relevance and context-recall are GNOMON v2, not built; no recall is promised.
- **Code symbol resolution is by unqualified, unique name** (ADR-G5) — high precision, limited recall, no type inference.
- **Token budget is char-estimated for retrieval**; generation tokens are real model counts.
- **n = 25**, single seed unless re-run; CI width reflects judge variance plus case-to-case variance.

## Reproduce

```bash
pip install -e ".[document,retrieval,embeddings,eval]"
export ANTHROPIC_API_KEY=... GROQ_API_KEY=...
make benchmark BOOK="<Monster Manual PDF>"   # writes METRICS.md + eval/benchmark-baseline.json
```
