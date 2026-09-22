# Supplemental XCOMET-XXL Significance Analysis

Final status: job **5798146 completed successfully in 00:02:05** after scorer
5798139. All 351 conditions and 846 planned comparisons are represented.
See [the final audit](../../grammar_context_evaluation/COMPLETION_AUDIT.md). Lezgi 87-row significance remains
reference-contaminated; the final 84/83-row sensitivities are descriptive only.

This analysis uses the existing sentence-level XXL scores. It loads no model,
uses no GPU, and makes no API calls. It must run after XXL scoring succeeds.
The current gold-support protocol and all previous analyses remain unchanged.

## Design Frozen Before XXL Results

- Validate all 351 conditions, source/reference alignment, saved prompt evidence,
  predictions, scorer/configuration hashes, COMET version and score provenance.
  Require one consistent XXL checkpoint across the matrix. Missing/stale scores
  fail the analysis; do not analyze a smaller successful-only intersection.
- Include empty predictions through their saved XXL scores; never drop failures.
- Compare each grammar condition with its same-model/method baseline. Also compare
  cheat sheet and summary-table JPG conditions against their TXT counterparts.
- Analyze both native cohorts and common99 (first 99 Tsez rows for every model).
- Use 100,000 paired sentence bootstrap samples: bootstrap the per-sentence score
  differences with replacement. Pair identities determine deterministic seeds
  derived from base seed 20260919. Identical native/common99 contrasts reuse the
  same computation rather than pretending to be independent replications.
- Compute two-sided p-values from the null-centered bootstrap distribution,
  using a plus-one correction and a 1e-12 floating-point tolerance. Zero observed
  differences receive p=1. This is an approximate bootstrap test, not a t-test,
  an exact test, or SacreBLEU's BLEU/chrF implementation.
- Report 95% percentile confidence intervals for the paired mean difference.
  These intervals are unadjusted; use Holm-adjusted p-values for significance.
- Apply Holm across **630 baseline/context tests** spanning both cohorts, and
  separately across **216 TXT/JPG tests** spanning both cohorts.
- These are supplemental XXL families, not a retrospective joint correction
  across XXL and the already defined BLEU/chrF++ families. Disclose that distinction
  and do not claim overall three-metric familywise control.
- Store native COMET scores, absolute paired differences, raw and adjusted p-values,
  seeds, intervals, input hashes, analysis hash and NumPy/SciPy versions.

## Outputs and Interpretation

`docs/matched_gold_v2/analysis_xcomet_xxl/` contains `scores.tsv`,
`comparisons.tsv`, `material_pairs.tsv`, `results.json` and `report.md`.

Positive delta means context exceeds baseline, or JPG exceeds TXT. A positive
effect with adjusted p<0.05 is evidence under this test, not proof of practical
importance, human adequacy, or improvement for every language. Nonsignificance
is not equivalence. No model-interaction test is included. Two cohorts overlap.
Sentence resampling does not capture generation-seed variability, and repeated
Lezgi examples are not handled with a cluster bootstrap. XXL language-coverage
limitations remain. TXT/JPG effects need content equivalence for a modality-only
interpretation.

## Resources and Local Checks

The job requests **one CPU, 24 GB RAM and one hour; zero GPUs**. A local synthetic
445-row comparison with 100,000 resamples took approximately 0.24 seconds. The
time limit leaves room for input verification, shared-filesystem I/O and cluster
load. The production analysis has now completed; this earlier benchmark is not
its measured elapsed time.

Four local tests passed: equal-score and known-shift cases, deterministic and
symmetric tests, Holm calculation and input mismatch rejection, and a full
351-condition synthetic integration run producing all 846 planned comparisons.
No production scores were created by those tests.

```bash
venv/bin/python -m unittest discover -s tests -p test_xxl_analysis.py -v
venv/bin/python runners/scoring/analyze_gold_v2_xcomet_xxl.py --dry-run
```

The dry-run validates configurations only. The actual analysis requires all
XXL artifacts to pass validation. See `xcomet_xxl_analysis_submission.json` for
the submitted job ID and dependency; do not submit a duplicate.
