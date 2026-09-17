# Matched Grammar-Context Results

Paired bootstrap: 100000 resamples. Cohort: common99.
Failures remain in the denominator. Incomplete experiments are excluded, not replaced by a smaller intersection.

| Model | Complete experiments | Planned experiments | Context contrasts | Holm-positive metric tests |
| --- | ---: | ---: | ---: | ---: |
| qwen3 | 117 | 117 | 105 | 1 |
| qwen35 | 117 | 117 | 105 | 0 |
| gemini25flashlite | 117 | 117 | 105 | 0 |
| gpt56luna | 56 | 117 | 44 | 0 |

This is a partial matrix whenever complete and planned counts differ. Luna may stop at the user-approved budget.
Use analysis_common99/scores.tsv for descriptive cross-model comparisons on identical source cohorts.
Native Tsez: Qwen3/Qwen3.5 use 445 rows; Gemini/Luna use the first 99 rows.
comparisons.tsv measures context minus the same-model, same-method baseline.
material_pairs.tsv tests TXT versus JPG; a representation-only interpretation additionally requires content equivalence.
No model-interaction significance or causal architecture claim is made. Local and API decoding policies differ.
Original GRAMMAMT prompt strategies are adapted, not reproduced verbatim. All methods retain the same first 21 support examples.
No significant difference is not evidence of equivalence. See results.json for hashes, missing conditions and limitations.
