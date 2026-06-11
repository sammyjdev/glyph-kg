# GraphRAG vs vector retrieval over a real corpus, measured

> **Validation-first draft (P6.2).** Every claim below points at the code or artifact that
> backs it. The results table is from a real benchmark run, committed to
> `eval/benchmark-baseline.json` and regenerable with `make benchmark`.

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
| Code KG, deterministic, Python + Java | `glyph/extract/code/`; self-hosted `out/glyph-code.json` (244 nodes, 384 edges) |
| Graph-aware retrieval + a fair vector baseline + hybrid, one output contract | `glyph/retrieval/`, `glyph/baseline/vector.py`, `glyph/model/contract.py`; ADR-G3 |
| Benchmark against GNOMON with bootstrap CIs | `glyph/eval/` (target, judge, harness, report); ADR-G4 |
| Frozen, reproducible query set | `eval/queries.json` (n=25), `scripts/build_query_set.py --check` |
| Reproducible baseline + regression gate | `make benchmark` / `make benchmark-check`, `eval/benchmark-baseline.json` |
| AXON consumes GLYPH as the graph source | `glyph/integration/GraphContextSource`; `docs/axon-integration.md` |

## Method (ADR-G4)

Three arms (graph, vector, hybrid) answer the same 25 questions over the same corpus, under the
same token budget and embedder. Each arm generates a grounded answer over its retrieved context
(real token + latency instrumentation). An OpenAI-compatible OSS judge (NVIDIA NIM,
`meta/llama-3.3-70b-instruct`) scores GNOMON's two v1 metrics — `faithfulness` and
`context_precision` — over `judge_runs=3`, and we aggregate per-case scores with a seeded
percentile bootstrap. Cost is generation tokens at Haiku 4.5 rates. The judge is transport-agnostic
(`--base-url`/`--api-key-env`), so any provider serving the same Llama 3.3 70B keeps the run comparable.

## Results

Run of 2026-06-11 — n=25, judge `meta/llama-3.3-70b-instruct`, `judge_runs=3`, seed 0. Cells are
**mean [95% percentile-bootstrap CI]**. Source: `eval/benchmark-baseline.json` / `METRICS.md`.

| Metric | graph | vector | hybrid |
|---|---|---|---|
| faithfulness | **0.987** [0.960, 1.000] | 0.928 [0.837, 0.995] | 0.933 [0.827, 1.000] |
| context_precision | 0.366 [0.236, 0.509] | 0.400 [0.200, 0.600] | **0.434** [0.251, 0.617] |
| total tokens | **30 831** | 35 074 | 42 882 |
| cost (US$, generation) | **0.0399** | 0.0433 | 0.0511 |
| mean latency (ms) | 1934 | 2089 | **1856** |

**Honest read.** The graph arm posts the **highest faithfulness** (0.987, tightest CI) at the
**lowest token cost** — ~12% fewer tokens than vector and ~28% fewer than hybrid. It does **not**
win `context_precision`: vector and hybrid are nominally higher there. But at n=25 every metric's
CIs overlap across arms, so no arm is *significantly* ahead on quality — the defensible claim is
that graph-aware retrieval **matches or beats** the vector baseline on both metrics while being the
**most token-efficient** arm. The hybrid buys the best `context_precision` and latency at the
highest token cost. Where the graph loses (`context_precision`) is the factual/attribute category,
exactly as anticipated — kept in the report rather than hidden.

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
export ANTHROPIC_API_KEY=...        # answer generation (Claude Haiku 4.5)
export NVIDIA_NIM_API_KEY=...        # OSS judge (NVIDIA NIM); Groq also works via --api-key-env
python3 scripts/run_benchmark.py out/monster-manual.json "<Monster Manual PDF>" \
  --base-url https://integrate.api.nvidia.com/v1 --api-key-env NVIDIA_NIM_API_KEY \
  --model meta/llama-3.3-70b-instruct   # writes METRICS.md + eval/benchmark-baseline.json
```
