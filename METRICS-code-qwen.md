# GLYPH benchmark — graph vs vector vs hybrid

- cases (n): **14**  ·  judge: `qwen/qwen3-next-80b-a3b-instruct`  ·  judge_runs: 3  ·  seed: 0
- metric cells show **mean [95% CI]** (percentile bootstrap via GNOMON).
- cost is generation only (Haiku 4.5 rates); judge tokens excluded. Tokens are real.

| Metric | graph | vector | hybrid |
|---|---|---|---|
| context_precision | 0.696 [0.515, 0.866] | 0.557 [0.282, 0.829] | 0.890 [0.829, 0.945] |
| faithfulness | 0.862 [0.712, 0.969] | 0.926 [0.867, 0.979] | 0.871 [0.748, 0.964] |
| total tokens | 14843 | 15507 | 14910 |
| cost (US$) | 0.0224 | 0.0252 | 0.0234 |
| mean latency (ms) | 13550.6 | 6418.1 | 9771.7 |

