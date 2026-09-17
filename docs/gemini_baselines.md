# Gemini no-grammar baselines

Submitted 2026-09-15. Each job runs baseline and ModelGloss baseline sequentially.

| Language | Test records per condition | Job |
| --- | --- | --- |
| Gitksan | 37 | 5790364 |
| Lezgi | 87 | 5790365 |
| Natugu | 99 | 5790366 |
| Tsez | 99 (first records, matching API subset) | 5790367 |

Model: google/gemini-2.5-flash-lite. Support examples: first 21 training records.
Temperature 0, seed 42, max output tokens 512, reasoning effort none.
Existing API runner, system message, parsing, and ModelGloss predictions are reused.
Only grammar context and its instruction are omitted. This is a few-shot baseline,
not a zero-shot baseline. Gitksan PDF1 and PDF2 share the baseline.

Jobs: scripts/jobs/openrouter/run_gemini25flashlite_baseline_<language>.sh
Outputs: results/baseline/gemini25flashlite/<language>/grammar/original/
Metrics: metrics/baseline/gemini25flashlite/<language>/grammar/original/
Each CPU job requests six hours. Planned requests: 644, excluding retries.
Existing results are resumed, not overwritten. No grammar experiment is rerun.

Basic metrics are produced by the runner. XCOMET and paired significance are
separate follow-up steps; they are not submitted by these generation jobs.
Only compare matching test prefixes and condition (baseline vs normal,
ModelGloss baseline vs ModelGloss), never full 445-record Tsez against this subset.
