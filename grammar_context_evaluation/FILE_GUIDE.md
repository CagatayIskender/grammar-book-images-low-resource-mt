# File Guide

## For Writing

| Question | File |
| --- | --- |
| What was tested and what can I claim? | [Thesis guide](../THESIS_GUIDE.md) |
| What corrections were made and why? | [Methods and limitations](../THESIS_GUIDE.md) |
| Are the main experiments complete? | [Completion audit](COMPLETION_AUDIT.md) |
| Which scores should I use? | [Recommended scores](thesis_scores.tsv) |
| Where is each condition's evidence? | [Condition catalog](condition_catalog.tsv) |
| Which figures can I use? | [Six-page matrix](figures/experiment_matrix_appendix.pdf) |
| What are the statistical conclusions? | [Statistical summary](significance_summary.tsv) and the detailed reports linked in [README](README.md) |
| What needs caution for Lezgi? | [Sensitivity scores](lezgi_sensitivity_scores.tsv) and the reference-quality section of the thesis guide |
| How were the files checked? | [Verification](VERIFICATION.md) and [input hashes](audit.json) |
| Where is the supplementary study? | [MTOB closure](../experiments/mtob/evaluation_v1/CLOSURE_REPORT.md) |

The main study contains 351 conditions. MTOB's 30 conditions are separate.
Historical experiments and smoke tests are not extra conditions in the main
matrix. Luna is excluded. Technical folder names outside grammar_context_evaluation/ preserve
recorded experiment identifiers and paths, not different versions of the final
score table.

## For Reproduction

Read the repository-root [README](../README.md) and the
[corrected protocol](../docs/matched_gold_v2/README.md). The existing runners
expect the source dataset under the sibling Database/2023glossingST/data path.
That external folder, virtual environments, cached model weights and credentials
are not part of this repository's publication. Configurations, saved predictions,
metric files and source hashes document the completed runs; a fresh checkout is
not a preconfigured cluster environment. Access to the original data and models
and installation of the recorded dependencies are required to rerun generation.

No API key, access token or private key is needed to read the thesis artifacts.
Do not add one to this package. Keep original language forms, references and
recorded outputs unchanged when preparing tables for the thesis.
