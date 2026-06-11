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

## Results — document domain (Monster Manual)

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

## Results — code domain (AXON source graph)

The same harness, pointed at a real code graph (AXON's `src/axon`, 1307 nodes) with a frozen
14-query set (7 structural — callers/inheritance — + 7 semantic, the latter authored and
oracle-verified by a second model). Generation ran on NVIDIA NIM (Llama 3.3 70B, free); the judge
is `gemini-2.5-flash` — **independent of the generator family on purpose**. Source:
`eval/code-benchmark-baseline.json` / `METRICS-code.md`.

| Metric | graph | vector | hybrid |
|---|---|---|---|
| faithfulness | 0.839 [0.682, 0.963] | **0.995** [0.988, 1.000] | 0.864 [0.699, 0.988] |
| context_precision | 0.180 [0.111, 0.266] | **0.513** [0.279, 0.737] | 0.353 [0.186, 0.531] |
| mean latency (ms) | 13551 | **6418** | 9772 |

**Honest read — the thesis did not hold here.** In the code domain the **fair vector baseline
beats graph-aware retrieval on both metrics** (faithfulness 0.995 vs 0.839; context_precision 0.513
vs 0.180), with hybrid in between. The graph arm anchors a symbol and expands two hops; the strict
independent judge penalises that neighbourhood as low-precision context, while the vector arm nails
the semantic queries. This is the validation-first rule paying its way: the comparison was built to
let either side win, and for *this* corpus + query set + judge, vector won. The graph axis still has
value (the document domain, and the global community axis of ADR-G7), but "graph beats vector for
code retrieval" is **not** supported by this run.

**A note on judges.** An earlier code run judged by a *Llama* model (same family as the generator)
rated the graph arm far more leniently (context_precision ≈ 0.59 vs Gemini's 0.18) — a self-evaluation
bias. The independent Gemini judge is the honest one; we report it. The two domains use different
judges, so compare arms *within* a domain, not absolute numbers *across* domains.

## Declared limitations

- **Relevance oracle is KG-derived, not gold.** It inherits extraction errors (e.g.
  `ankheg → resists ácido`), kept on purpose so the comparison exposes graph noise.
- **Two metrics only.** Answer-relevance and context-recall are GNOMON v2, not built; no recall is promised.
- **Code symbol resolution is by unqualified, unique name** (ADR-G5) — high precision, limited recall, no type inference.
- **Token budget is char-estimated for retrieval**; generation tokens are real model counts.
- **n = 25** (documents) / **n = 14** (code), single seed unless re-run; CI width reflects judge variance plus case-to-case variance.
- **Code vector arm is per-file**, while the graph arm is per-symbol — a declared granularity asymmetry the run measures rather than corrects.
- **Different judges across domains** (documents: Llama 3.3 70B; code: Gemini 2.5 Flash), so arm comparisons are valid *within* a domain, not as absolute numbers *across* domains.

## Reproduce

```bash
pip install -e ".[document,retrieval,embeddings,eval]"
export ANTHROPIC_API_KEY=...        # answer generation (Claude Haiku 4.5)
export NVIDIA_NIM_API_KEY=...        # OSS judge (NVIDIA NIM); Groq also works via --api-key-env
python3 scripts/run_benchmark.py out/monster-manual.json "<Monster Manual PDF>" \
  --base-url https://integrate.api.nvidia.com/v1 --api-key-env NVIDIA_NIM_API_KEY \
  --model meta/llama-3.3-70b-instruct   # writes METRICS.md + eval/benchmark-baseline.json

# code domain — generation on NIM (free), independent Gemini judge:
export GEMINI_API_KEY=...
python3 scripts/run_benchmark.py out/axon-code.json <repo>/src/axon --domain code \
  --gen-base-url https://integrate.api.nvidia.com/v1 --gen-api-key-env NVIDIA_NIM_API_KEY \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai \
  --api-key-env GEMINI_API_KEY --model gemini-2.5-flash --judge-no-seed
```
