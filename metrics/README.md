# Metrics

Each score file has one location. There are no alternative directory views or
symlinks. Follow the same experiment family in configs/, results/ and docs/.

## Current Study: Version 2

| Directory | Contents |
| --- | --- |
| [matched_gold_v2/lexical_and_xcomet_xl](matched_gold_v2/lexical_and_xcomet_xl/) | Corrected 351-condition study: BLEU, chrF++ and XCOMET-XL |
| [matched_gold_v2/xcomet_xxl](matched_gold_v2/xcomet_xxl/) | XCOMET-XXL for those same 351 conditions; not another experiment version |

Both collections use 21 nonempty gold-glossed training supports. For writing,
use [reports/matched_gold_v2/thesis_scores.tsv](../reports/matched_gold_v2/thesis_scores.tsv)
and the [reporting policy](../reports/matched_gold_v2/README.md).
Lezgi's filtered scores are descriptive, not newly tested significance results.

## Earlier Work: Do Not Pool with Version 2

| Directory | Meaning |
| --- | --- |
| [matched_v1](matched_v1/) | Earlier matched protocol with empty support gloss fields |
| [chain_gloss_v2](chain_gloss_v2/) | Earlier chain-gloss repair family; its v2 is local to that family, not matched_gold_v2 |
| [curated_v1](curated_v1/) | Earlier curated-material experiments, not the corrected matched study |
| [baseline](baseline/) | Earlier baselines; current matched baselines are inside matched_gold_v2 |
| [legacy](legacy/) | Earlier grammar-context experiments |
| [smoke](smoke/) | Small API pilot tests, not full test sets |

The historical MTOB comparison uses matched_v1. It does not compare against the
corrected version-2 baselines. Its scores and reports remain under experiments/mtob/.

## Naming Rule

Names identify an experiment family and, where applicable, its protocol version.
Do not interpret v1/v2 belonging to different families as a global chronology.
The subdirectory xcomet_xxl identifies an evaluator, not a new protocol.
Model/language/source/variant subdirectories retain the saved experimental IDs.
See [repository structure](../docs/REPOSITORY_STRUCTURE.md) for the full map.

Recorded configurations and historical analysis JSON retain their original
metric-path strings to preserve their fingerprints and provenance. Active tools
resolve those strings using runners/artifact_layout.py; the
[migration manifest](../docs/metric_layout_migration.json) records the 702
old/new paths and unchanged file hashes. There are no compatibility symlinks.
