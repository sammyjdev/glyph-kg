# GLYPH benchmark — graph vs vector vs hybrid

- cases (n): **8**  ·  judge: `gemini-2.5-flash`  ·  judge_runs: 3  ·  seed: 0
- metric cells show **mean [95% CI]** (percentile bootstrap via GNOMON).
- cost is generation only (Haiku 4.5 rates); judge tokens excluded. Tokens are real.

| Metric | community | vector | graph |
|---|---|---|---|
| context_precision | 0.794 [0.600, 0.950] | 0.694 [0.375, 1.000] | 0.532 [0.287, 0.782] |
| faithfulness | 0.927 [0.794, 1.000] | 0.994 [0.981, 1.000] | 0.952 [0.885, 1.000] |
| total tokens | 5337 | 10148 | 10355 |
| cost (US$) | 0.0133 | 0.0212 | 0.0229 |
| mean latency (ms) | 45140.2 | 46872.2 | 55452.3 |

