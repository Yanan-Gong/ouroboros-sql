### Eval: `val` split (ablation-val-memory-v1)

63 examples (3 adversarial probes) x 4 repeats = 252 records (0 harness errors)

| Metric | Value [95% CI] |
|---|---|
| Execution accuracy (A_mean) | 53.8 [42.9, 64.6] |
| Aptitude (A90) | 63.5 [51.3, 75.2] |
| Worst-case (A10) | 40.3 [29.3, 51.7] |
| Unreliability (U90) | 23.2 [14.5, 32.3] |
| Judge score (mean) | 75.8 [70.5, 81.2] |
| Judge unreliability (U90) | 12.3 [8.1, 16.6] |
| Refusal accuracy (adversarial) | 75.0 [25.0, 100.0] |
| False-refusal rate | 0.0 [0.0, 0.0] |
| Schema-grounding precision | 93.7 [90.1, 96.7] |
| Schema-grounding recall | 96.9 [94.1, 99.2] |
| Wasted execute_sql rate | 0.0 [0.0, 0.0] |
| Routing accuracy | 100.0 [100.0, 100.0] |
| Completion rate | 87.9 [82.1, 92.9] |
| Handoff ping-pong (mean count) | 0.0 [0.0, 0.0] |
| Judge-exec agreement | 55.8 |
| Tokens per question (in+out) | 41323+2798 |
| Model calls per question | 11.2 |
| Latency p50 / p95 (s) | 75 / 146 |
