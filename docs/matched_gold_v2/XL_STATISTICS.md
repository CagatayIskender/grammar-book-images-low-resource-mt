# Supplemental XCOMET-XL Significance Analysis

Job **5798172 completed successfully in 00:01:54**, after XL scorer 5798036.
It used one CPU and no GPU. The BLEU/chrF++ and XXL analyses also completed.
Use [the final audit](../../thesis/COMPLETION_AUDIT.md); do not submit a duplicate.

The XL scorer was still pending when updated to retain sentence scores under
`xcomet_segments.translation` for matched_gold_v2. Aggregate-only metrics cannot
support a paired sentence analysis. The same scoring pass now records segments,
checkpoint/scorer hashes and COMET version; no additional GPU job is needed.
The model remains Unbabel/XCOMET-XL and the mean-score calculation is unchanged.
No translation, configuration or historical metric file was changed by this edit.

The XL loader verifies all 351 conditions, actual prompts, source/reference
alignment, configuration fingerprints and prediction-bound segment scores. It
requires the same checkpoint throughout and checks that the saved mean matches
its indexed segments. Incomplete, stale or nonfinite scores fail validation.
The scorer will not reuse aggregate-only corrected metrics as segment-complete.

XL and XXL share the same tested statistical computation: 100000 null-centered
paired bootstrap resamples, two-sided plus-one p-values, 95% unadjusted
paired-difference intervals, native/common99 cohorts, and separate Holm families
of 630 baseline/context and 216 TXT/JPG tests across cohorts. Each metric has its
own supplemental families. There is no joint adjustment across XL, XXL, BLEU and
chrF++; do not select whichever metric gives significance and claim overall
familywise control. Identical cohorts are not independent replications.

XL outputs are separate: `docs/matched_gold_v2/analysis_xcomet_xl/` contains
`scores.tsv`, `comparisons.tsv`, `material_pairs.tsv`, `results.json`, and
`report.md`. Score columns identify XL; no XXL scores are imported. Absolute
deltas use native COMET scale. Empty predictions remain through their XL scores.
The limitations in [XXL statistics](XXL_STATISTICS.md) also apply to XL.

Eight local XL/XXL analysis tests passed, including stale/aggregate-only XL
rejection, segment-index/mean checks and the full synthetic comparison matrix.
The completed production analysis validates all 351 conditions. Synthetic tests
alone did not establish production correctness; final input hashes and score
checks are recorded separately. Full-cohort Lezgi tests contain three invalid
references and are not confirmatory findings; use final/README.md for sensitivities.
