# GLYPH benchmark — graph vs vector vs hybrid

- cases (n): **25**  ·  judge: `deepseek/deepseek-chat`  ·  seed: 0
- metric cells show **mean [95% CI]** (percentile bootstrap, numpy, 2000 resamples).
- cost is generation only (Haiku 4.5 rates); judge tokens excluded. Tokens are real.

| Metric | graph | graph_pagerank | vector | hybrid | multi_anchor | path | reranked_vector |
|---|---|---|---|---|---|---|---|
| context_precision | 0.562 [0.380, 0.746] | 0.590 [0.400, 0.780] | 0.400 [0.200, 0.600] | 0.624 [0.444, 0.800] | 0.504 [0.340, 0.690] | 0.030 [0.000, 0.080] | 0.720 [0.560, 0.880] |
| faithfulness | 0.980 [0.940, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.920 [0.820, 0.989] | 0.960 [0.900, 1.000] | 0.980 [0.940, 1.000] |
| total tokens | 31600 | 31435 | 30995 | 39968 | 40869 | 6082 | 33195 |
| cost (US$) | 0.0578 | 0.0576 | 0.0461 | 0.0640 | 0.0704 | 0.0152 | 0.0534 |
| mean latency (ms) | 7594.1 | 9457.5 | 9051.9 | 11496.5 | 12395.3 | 1827.0 | 427684.7 |

