# Final Completion and Comparability Audit

Verified: 2026-09-22T18:52:55.735395+00:00. Scope: matched_gold_v2 only.

All 351 conditions / 40,731 records pass complete-index, source/reference, frozen configuration,
21 gold-support and saved actual system/user-message verification. Both XL and XXL artifacts
match these predictions. Saved analysis input hashes and TSV/JSON contents agree.
Basic BLEU and chrF++ were recomputed and matched. No model was loaded; no API or generation was run.

## Status by Model

| Model | Conditions | Records | Empty | Truncated | Missing chain gloss |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | 117 | 15999 | 197 | 179 | 166 |
| qwen35 | 117 | 15999 | 20 | 20 | 14 |
| gemini25flashlite | 117 | 8733 | 372 | 359 | 338 |

## Statistical Output Completeness

| Analysis | Family | Tests | Significant positive | Significant negative |
| --- | --- | ---: | ---: | ---: |
| native | comparisons | 630 | 1 | 10 |
| native | material_pairs | 216 | 1 | 9 |
| common99 | comparisons | 630 | 0 | 3 |
| common99 | material_pairs | 216 | 1 | 4 |
| xcomet_xl | comparisons | 630 | 0 | 4 |
| xcomet_xl | material_pairs | 216 | 2 | 0 |
| xcomet_xxl | comparisons | 630 | 0 | 1 |
| xcomet_xxl | material_pairs | 216 | 0 | 1 |

Counts are tests, not independent discoveries. Cohorts overlap; correction families are separate.
Full-cohort Lezgi tests include invalid references and must not establish a primary improvement claim.
No significant difference is not equivalence. No cross-model interaction is tested.

## Reference-Quality Decision

The frozen 87-row Lezgi analyses are preserved as protocol outputs, not silently replaced.
Use the 84-row valid-reference sensitivity for descriptive Lezgi performance; indices 37, 62 and 81
are excluded solely because their source-dataset references are literal `nan`.
The 83-row sensitivity additionally removes index 11, whose reference occurs in a longer training support.
The same index masks apply to every baseline/context/model/method; empty predictions remain included.
Scores are recomputed from the same predictions. XL/XXL use the corresponding saved sentence scores.
These post hoc sensitivity results are descriptive only: no filtered-cohort significance or universal gain is claimed.
Partial training/test overlap, duplicate rows and unassessed book/test overlap remain limitations.

## Readiness

**Share with caveats.** Production, scoring and planned tests are complete. No generation job remains
necessary for this frozen study. Write findings as effects of this adapted context package, not exact
GRAMMAMT replication or proven improvements across languages. Lezgi conclusions are descriptive.
See audit.json for scope, hashes and remaining limits; see README.md for handover navigation.
