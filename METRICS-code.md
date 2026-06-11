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

