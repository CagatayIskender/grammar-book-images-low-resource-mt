# Metric Collections

## Current Study: Gold-Glossed Supports

Use these readable local directory views for the corrected 351-condition study:

| Directory | Contents | Canonical files (also browsable on GitHub) |
| --- | --- | --- |
| gold_gloss_supports/bleu_chrf_xcomet_xl | BLEU, chrF++ and XCOMET-XL | [matched_gold_v2](matched_gold_v2/) |
| gold_gloss_supports/xcomet_xxl | Separate XCOMET-XXL scoring | [matched_gold_v2_xcomet_xxl](matched_gold_v2_xcomet_xxl/) |

Subdirectories retain model, language, grammar source and original/Cyrillic
variant. The current study uses 21 nonempty gold-glossed training supports.
For writing, use the [recommended score table](../grammar_context_evaluation/thesis_scores.tsv)
and its [evaluation policy](../grammar_context_evaluation/README.md), not an
unqualified merge of all JSON files. Lezgi's filtered scores are descriptive.

## Historical and Pilot Studies

| Readable directory | Canonical files | Status |
| --- | --- | --- |
| historical_studies/empty_support_glosses | [matched_v1](matched_v1/) | Earlier matched study with empty support gloss fields; excluded from the primary study |
| historical_studies/earlier_grammar_materials | [curated_v1](curated_v1/) | Earlier material experiments |
| historical_studies/earlier_chain_gloss | [chain_gloss_v2](chain_gloss_v2/) | Earlier explicit chain-gloss experiments |
| historical_studies/earlier_baselines | [baseline](baseline/) | Earlier baseline experiments |
| historical_studies/legacy_grammar_runs | [legacy](legacy/) | Legacy grammar-context experiments |
| pilot_tests/api_smoke_tests | [smoke](smoke/) | Pilot tests, not full primary-study conditions |

Historical results are retained for provenance, not pooled with the current study.
In particular, matched_v1 must not be described as using gold-glossed supports.
Its MTOB comparison remains a historical, cross-protocol analysis.

## Why Canonical Technical Names Remain

The readable directories are relative symbolic links, not duplicate files.
Frozen configurations and validators bind outputs to their original paths and
hashes. Moving those physical directories would break validation; rewriting
the frozen records would change the experimental evidence. The original
technical names therefore remain as canonical storage paths.

On GitHub, use the canonical links in the tables if a symbolic link is shown as
a text file. To recreate the views locally, run from the repository root:

```bash
python3 scripts/organize_metric_views.py
```

These views do not change scientific results or create new experiments. When
collecting metrics, do not follow both a canonical directory and its view, or
the same condition will be counted twice.
