# GLYPH benchmark — graph vs vector vs hybrid

- cases (n): **14**  ·  judge: `gemini-2.5-flash`  ·  judge_runs: 3  ·  seed: 0
- metric cells show **mean [95% CI]** (percentile bootstrap via GNOMON).
- cost is generation only (Haiku 4.5 rates); judge tokens excluded. Tokens are real.

| Metric | graph | vector | hybrid |
|---|---|---|---|
| context_precision | 0.180 [0.111, 0.266] | 0.513 [0.279, 0.737] | 0.353 [0.186, 0.531] |
| faithfulness | 0.839 [0.682, 0.963] | 0.995 [0.988, 1.000] | 0.864 [0.699, 0.988] |
| total tokens | 14843 | 15507 | 14910 |
| cost (US$) | 0.0224 | 0.0252 | 0.0234 |
| mean latency (ms) | 13550.6 | 6418.1 | 9771.7 |


## 2026-07-26 — external benchmark (graphify-vs-glyph) and the granularity fix

An external pre-registered benchmark (harness-bench/graphify-vs-glyph:
encode/httpx corpus, N=30 owner-validated stratified cases, 3-judge GNOMON
panel, CI-separation win criteria) measured the whole-file vector corpus as
a granularity handicap and drove #50/#51/#52. context_precision, pooled,
mean [95% CI] per judge (Llama-3.1-8B / GLM-4.6 / Mistral-Nemo):

| Arm | Llama | GLM | Mistral |
|---|---|---|---|
| vector (whole-file) | 0.25 [0.11,0.41] | 0.20 [0.08,0.35] | 0.42 [0.23,0.60] |
| vector-sym (#51) | 0.87 [0.79,0.93] | 0.68 [0.54,0.81] | 0.73 [0.59,0.87] |
| hybrid (whole-file) | 0.52 [0.37,0.68] | 0.22 [0.11,0.36] | 0.39 [0.24,0.55] |
| hybrid-sym | 0.67 [0.55,0.77] | 0.70 [0.62,0.78] | 0.85 [0.77,0.92] |
| graph (unchanged) | 0.53 [0.38,0.66] | 0.62 [0.48,0.75] | 0.74 [0.61,0.86] |
| graphify (95k-star external) | 0.69 [0.55,0.83] | 0.72 [0.63,0.81] | 0.97 [0.94,1.00] |

Findings on record:

- `code_symbol_documents` beats the whole-file corpus CI-separated, 3/3
  judges, every stratum — and is indistinguishable from graphify on every
  stratum. On code, the retrieval unit dominated the retrieval strategy.
- **Hybrid guard (#52): hybrid must not sit CI-separated below its best
  parent by judge majority.** Status 2026-07-26: PASSES with symbol
  granularity (1/3 judges rank vector-sym above it; majority required).
  Whole-file hybrid FAILED this guard (lost to its own graph parent 2/3).
- Honest limit: hybrid-sym does not beat its parents either — fusion adds
  no measurable value at aligned granularity on this corpus. Do not grow
  the hybrid path without new evidence.
