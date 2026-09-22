# Grammar-Context Evaluation

Primary study: **matched_gold_v2**, Qwen3, Qwen3.5 and Gemini; Luna excluded.
All 351 conditions, XL/XXL scores and planned statistical analyses completed.
Readiness: **share with caveats**, not a claim of universal improvement.

New to the repository? Start with the short [file guide](FILE_GUIDE.md).
Raw scores are indexed in [Metric collections](../metrics/README.md), with
separate readable views for the corrected study and historical empty-gloss runs.

## Reading Order

1. [Thesis guide](../THESIS_GUIDE.md): design, corrections, methods and limitations.
2. [Completion and comparability audit](COMPLETION_AUDIT.md): checked inputs and scope.
3. [Overview JPG](figures/experiment_matrix_overview.jpg) / [PDF](figures/experiment_matrix_overview.pdf).
4. [Detailed 351-condition JPG](figures/experiment_matrix_detailed.jpg) / [PDF](figures/experiment_matrix_detailed.pdf).
   For printing, use the [six-page appendix matrix](figures/experiment_matrix_appendix.pdf).
5. [Condition catalog](condition_catalog.tsv): exact result/config/metric paths and failure counts.
6. [Recommended thesis scores](thesis_scores.tsv): explicit cohort, reference policy and inference status.
   [Original full-cohort scores](scores.tsv) preserve the 87-row Lezgi protocol outputs separately.
7. [Statistical summary](significance_summary.tsv): test counts, not independent discoveries.
8. [Audit manifest](audit.json): SHA256 values and validation boundaries.

See [verification details](VERIFICATION.md) for tests and rendered checks.
Methodological corrections are documented in the thesis guide.

The figures display record coverage, not translation accuracy. Empty responses
are shown explicitly. This is the corrected matrix; historical matched_v1 images
do not describe these results. PDF figures are suitable for inclusion in a thesis;
the same figures are available as JPG for quick viewing.

In scores.tsv, evaluated_records identifies the scoring cohort; records,
expected and failure-count metadata describe the full generated condition.
Do not divide full-condition failure counts by the smaller common99 denominator.
BLEU/chrF++ use a 0-100 scale; XL and XXL retain their native COMET scales.

## Statistical Results

- [Native BLEU/chrF++](../docs/matched_gold_v2/analysis_native/report.md) and [comparisons](../docs/matched_gold_v2/analysis_native/comparisons.tsv).
- [Common99 BLEU/chrF++](../docs/matched_gold_v2/analysis_common99/report.md) and [comparisons](../docs/matched_gold_v2/analysis_common99/comparisons.tsv).
- [XL](../docs/matched_gold_v2/analysis_xcomet_xl/report.md) and [comparisons](../docs/matched_gold_v2/analysis_xcomet_xl/comparisons.tsv).
- [XXL](../docs/matched_gold_v2/analysis_xcomet_xxl/report.md) and [comparisons](../docs/matched_gold_v2/analysis_xcomet_xxl/comparisons.tsv).

Each directory also contains material_pairs.tsv and results.json. Use adjusted
p-values under each declared family. Do not merge families, count overlapping
cohorts as replications, or interpret nonsignificance as equivalence. Lexical
confidence intervals refer to individual system scores; COMET intervals refer
to paired differences. Neither measures generation-seed uncertainty.

## Lezgi: Explicit Evaluation Policy

[Sensitivity scores](lezgi_sensitivity_scores.tsv) and
[paired descriptive deltas](lezgi_sensitivity_contrasts.tsv) supplement rather
than overwrite the original 87-row artifacts.

| Cohort | Retained rows | Exclusion rule |
| --- | ---: | --- |
| Frozen protocol | 87 | Original recorded test set; includes invalid references |
| valid_reference84 | 84 | Remove source-dataset literal nan references at indices 37, 62, 81 |
| valid_reference_no_overlap83 | 83 | Also remove index 11, whose reference occurs inside a training support translation |

Indices are zero-based. Rules depend on data quality, not predictions or scores.
All models/methods/materials use identical masks; failed predictions within the
retained set stay in the denominator. No missing reference is invented. Lexical
metrics are recomputed; XL/XXL average matching stored sentence scores.

**Use 84-row values for descriptive Lezgi reporting and show the 83-row check.**
Do not claim that 87-row Lezgi p-values establish improvement, or transfer them
to filtered cohorts. These post hoc sensitivity tables have no new significance
tests. This is a deliberate reporting restriction, not an unsubmitted experiment.
Duplicates and possible book/test overlap remain limitations.

## Safe Conclusions

The study measures the effect of grammar-context packages under an adapted
GRAMMAMT protocol. It does not demonstrate a general, consistent performance gain.
An insignificant contrast may reflect limited precision; it does not prove no
effect. Cross-model results are descriptive, not architecture-causal evidence.
TXT/JPG effects may include content/layout differences, not just modality.

## Rebuild

```bash
venv/bin/python scripts/finalize_gold_v2.py
venv/bin/python scripts/render_gold_v2_appendix.py
venv/bin/python -m unittest discover -s tests -p test_final_gold_v2.py -v
```

The finalizer loads no model and makes no network/API call. It does not alter
frozen prompts, results, metrics or statistical artifacts. It checks file
identities and writes this directory's derived tables, audit and figures.
Only small verification tests use synthetic data; delivered tables use real results.

Historical MTOB results remain a separate supplement. The existing MTOB/GRAMMAMT
comparison uses matched_v1, not these corrected baselines. Rebuild commands do
not publish anything. Do not package secrets, environments or caches.
