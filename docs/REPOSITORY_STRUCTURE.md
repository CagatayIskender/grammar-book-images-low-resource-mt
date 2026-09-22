# Repository Structure

## Start Here

1. Read [THESIS_GUIDE.md](../THESIS_GUIDE.md) for methods and limitations.
2. Open [reports/matched_gold_v2](../reports/matched_gold_v2/README.md) for the current tables and figures.
3. Use its [condition catalog](../reports/matched_gold_v2/condition_catalog.tsv) to locate individual configurations, predictions and scores.

## One Location per Artifact

The repository uses an artifact-based layout, with experiment-family identifiers
shared across directories. This is a common research-project convention, not a
claim that there is a universal mandatory naming standard.

```text
GRAMMAMT/
  README.md                 Entry point
  THESIS_GUIDE.md           Scientific methods and reporting restrictions
  reports/
    matched_gold_v2/       Current primary-study tables, figures and audit
  configs/                 Recorded experiment settings
  results/                 Model responses
  metrics/                 Per-condition scores
  docs/                    Protocols, statistical analyses and provenance
  inputs/                  Original grammar PDFs
  materials/               Derived grammar contexts
  runners/                 Generation and evaluation code
  scripts/                 Jobs, submissions and utilities
  tests/                   Verification tests
  experiments/mtob/        Isolated supplementary study
  archive/                 Superseded or unclassified historical artifacts
  slurm_outputs/           Local scheduler logs, not publication data
  venv/                    Local environment, not publication data
```

## Versions and Families

| Identifier | Meaning | Use for the primary thesis? |
| --- | --- | --- |
| matched_v1 | Earlier matched study; training support glosses were empty | No; historical evidence only |
| matched_gold_v2 | Corrected matched study; 21 nonempty gold training glosses | Yes, with the thesis guide's qualifications |
| matched_gold_v2/xcomet_xxl | Additional evaluator under the same version-2 metric directory | Yes; not another experiment family |
| curated_v1 | Earlier curated-material experiment family | Historical only |
| chain_gloss_v2 | Earlier chain-gloss repair family | Historical only; not version 2 of the final matched study |
| baseline / legacy / smoke | Earlier baselines, older contexts, API pilots | Historical only |

Version numbers belong to their named family. A material version is not a
protocol version. For example, current matched_gold_v2 experiments can correctly
use materials/curated_v1 without becoming the earlier curated_v1 experiment.
Current no-book baselines live inside matched_gold_v2, not metrics/baseline.

For the current study, follow the same ID in:

```text
configs/matched_gold_v2/       Frozen settings
results/matched_gold_v2/       Saved responses
metrics/matched_gold_v2/
  lexical_and_xcomet_xl/      BLEU, chrF++ and XCOMET-XL
  xcomet_xxl/                XCOMET-XXL
docs/matched_gold_v2/          Protocol and statistical evidence
reports/matched_gold_v2/       Joined tables, figures and completion audit
```

## September 2026 Navigation Cleanup

| Previous location | Current location or action |
| --- | --- |
| grammar_context_evaluation/ | Moved to reports/matched_gold_v2/ |
| metrics/gold_gloss_supports/ | Removed redundant links; use the two real subdirectories under metrics/matched_gold_v2 |
| metrics/matched_gold_v2/<model>/ | Moved under metrics/matched_gold_v2/lexical_and_xcomet_xl/<model>/ |
| metrics/matched_gold_v2_xcomet_xxl/ | Moved to metrics/matched_gold_v2/xcomet_xxl/ |
| metrics/historical_studies/ | Removed redundant links; use the actual family directories listed above |
| metrics/pilot_tests/ | Removed redundant link; use metrics/smoke |
| scripts/organize_metric_views.py | Removed; alternate views must not be regenerated |

There are no duplicate metric-directory aliases. Scientific identifiers,
configuration bytes, responses, scores and historical statistical outputs remain
unchanged. Frozen files retain their originally recorded path strings; active
tools translate those strings through runners/artifact_layout.py. The
[migration manifest](metric_layout_migration.json) records 702 file relocations
with hashes. Original production-code snapshots and approved path-only runtime
adapters are validated separately. This compatibility is in code, not extra
folders or symlinks, so every metric file has exactly one physical location.
No new experiment version is created by this navigation cleanup. Avoid
final/latest/updated as version names; use the family and protocol version plus
a Git commit.
