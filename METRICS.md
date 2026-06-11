# GLYPH benchmark — graph vs vector vs hybrid

- cases (n): **25**  ·  judge: `meta/llama-3.3-70b-instruct`  ·  judge_runs: 3  ·  seed: 0
- metric cells show **mean [95% CI]** (percentile bootstrap via GNOMON).
- cost is generation only (Haiku 4.5 rates); judge tokens excluded. Tokens are real.

| Metric | graph | vector | hybrid |
|---|---|---|---|
| context_precision | 0.366 [0.236, 0.509] | 0.400 [0.200, 0.600] | 0.434 [0.251, 0.617] |
| faithfulness | 0.987 [0.960, 1.000] | 0.928 [0.837, 0.995] | 0.933 [0.827, 1.000] |
| total tokens | 30831 | 35074 | 42882 |
| cost (US$) | 0.0399 | 0.0433 | 0.0511 |
| mean latency (ms) | 1934.3 | 2088.7 | 1855.7 |

