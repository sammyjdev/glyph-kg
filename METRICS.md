# GLYPH benchmark — graph vs vector vs hybrid

- cases (n): **25**  ·  judge: `deepseek/deepseek-chat`  ·  seed: 0
- metric cells show **mean [95% CI]** (percentile bootstrap, numpy, 2000 resamples).
- cost is generation only (Haiku 4.5 rates); judge tokens excluded. Tokens are real.

| Metric | graph | vector | hybrid | multi_anchor | path |
|---|---|---|---|---|---|
| context_precision | 0.435 [0.290, 0.585] | 0.460 [0.280, 0.640] | 0.347 [0.231, 0.476] | 0.338 [0.208, 0.477] | 0.040 [0.000, 0.100] |
| faithfulness | 0.992 [0.979, 1.000] | 1.000 [1.000, 1.000] | 0.943 [0.843, 1.000] | 0.929 [0.838, 0.986] | 1.000 [1.000, 1.000] |
| total tokens | 27317 | 30274 | 37840 | 35977 | 3171 |
| cost (US$) | 0.0324 | 0.0357 | 0.0432 | 0.0409 | 0.0045 |
| mean latency (ms) | 4720.5 | 4617.0 | 4319.3 | 4155.6 | 2575.8 |

