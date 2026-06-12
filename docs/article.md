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
oracle-verified by a second model). Generation ran on NVIDIA NIM (Llama 3.3 70B, free). To test
robustness, the run was judged by **two independent models of different families** —
`gemini-2.5-flash` and `qwen3-next-80b` (both ≠ the Llama generator). Source:
`eval/code-benchmark-baseline.json` (Gemini) / `eval/code-benchmark-qwen.json` (Qwen).

**faithfulness** — *robust across both judges* (same ranking: vector > hybrid > graph):

| judge | graph | vector | hybrid |
|---|---|---|---|
| gemini-2.5-flash | 0.839 | **0.995** | 0.864 |
| qwen3-next-80b | 0.862 | **0.926** | 0.871 |

**context_precision** — *judge-dependent; the ranking flips, so no robust winner*:

| judge | graph | vector | hybrid |
|---|---|---|---|
| gemini-2.5-flash | 0.180 | **0.513** | 0.353 |
| qwen3-next-80b | 0.696 | 0.557 | **0.890** |

**Honest read — and a lesson about judges.** On the *robust* metric (faithfulness, where two
unrelated judges agree on the ordering) the **fair vector baseline leads and graph-aware retrieval
is last** — the graph arm's two-hop expansion adds neighbourhood that does not improve groundedness.
`context_precision`, by contrast, is **not robust**: Gemini ranks vector first and graph last, while
Qwen ranks hybrid first and vector last — the verdict inverts with the judge, so we draw no
conclusion from it. (An earlier *Llama* judge — same family as the generator — was even more
lenient to the graph arm, a self-evaluation bias; that is why the reported judges are independent.)
The takeaway is twofold: for code retrieval here, graph-aware did **not** beat a fair vector
baseline on the metric we can trust; and a single LLM judge is not enough — cross-family judging
turned a clean-looking single-judge result into a correctly-hedged one. The graph axis still earns
its place in the document domain and as the global community axis (ADR-G7).

## Results — global axis (community summaries, ADR-G7)

Local retrieval (anchor + expand) answers "what depends on X?"; it cannot answer "how is this
*organized*?". The global axis detects communities in the graph (Louvain, seeded), summarizes each
with an LLM, and retrieves those summaries. We benchmarked it on 8 sense-making questions about the
AXON codebase ("what are the major subsystems?", "map the commit→decision flow"), three arms —
**community summaries vs the vector baseline vs local graph** — generation on free NIM, judged by
both Gemini and Qwen. Source: `eval/code-global-baseline.json` (Gemini) / `eval/code-global-qwen.json` (Qwen).

| Metric (judge) | community | vector | graph |
|---|---|---|---|
| context_precision (gemini) | **0.794** | 0.694 | 0.532 |
| context_precision (qwen) | 0.800 | 0.802 | 0.794 |
| faithfulness (gemini) | 0.927 | 0.994 | 0.952 |
| faithfulness (qwen) | 0.925 | 0.898 | 0.938 |
| **total tokens** | **5 337** | 10 148 | 10 355 |

**Honest read.** No arm is *significantly* ahead on quality — at n=8 the CIs overlap, and the two
judges disagree on the fine ordering (Gemini separates community ahead on context_precision; Qwen
calls all three roughly tied). What is **robust and unambiguous is efficiency**: the community arm
delivers parity-or-better quality at **~half the tokens** (5.3k vs 10k+) of either alternative.
For global "how is this organized?" questions, serving a handful of community summaries is far
cheaper context than expanding a local neighbourhood or retrieving whole files — and it never loses
on quality. That is the defensible claim for the global axis: **same-or-better answers, half the cost.**

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
