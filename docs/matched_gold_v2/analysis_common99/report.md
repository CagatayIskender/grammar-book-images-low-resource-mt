# Matched Grammar-Context Results

Paired bootstrap: 100000 resamples. Cohort: common99.
Failures remain in the denominator. Incomplete experiments are excluded, not replaced by a smaller intersection.

| Model | Complete experiments | Planned experiments | Context contrasts | Holm-positive metric tests |
| --- | ---: | ---: | ---: | ---: |
| qwen3 | 117 | 117 | 105 | 0 |
| qwen35 | 117 | 117 | 105 | 0 |
| gemini25flashlite | 117 | 117 | 105 | 0 |

This is a partial matrix whenever complete and planned counts differ. Only Qwen3, Qwen3.5 and Gemini are included.
Use analysis_common99/scores.tsv for descriptive cross-model comparisons on identical source cohorts.
Native Tsez: Qwen3/Qwen3.5 use 445 rows; Gemini uses the first 99 rows. Luna is excluded.
comparisons.tsv measures context minus the same-model, same-method baseline.
material_pairs.tsv tests TXT versus JPG; a representation-only interpretation additionally requires content equivalence.
No model-interaction significance or causal architecture claim is made. Local and API decoding policies differ.
Original GRAMMAMT prompt strategies are adapted, not reproduced verbatim. All methods share 21 gold-glossed supports; Lezgi training index 5 was replaced by index 21 before generation.
No significant difference is not evidence of equivalence. See results.json for hashes, missing conditions and limitations.

Final reporting qualification: the 87-row Lezgi analyses include three literal nan references.
They are retained protocol results, not confirmatory Lezgi evidence. See ../final/README.md
for 84/83-row descriptive sensitivities. Numerical statistical artifacts are unchanged.
