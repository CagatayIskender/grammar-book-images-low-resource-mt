# Grammar-Context Translation Experiments

This repository studies how grammar context affects Gloss-shot, Chain-gloss
and ModelGloss translation. The current primary study is **matched_gold_v2**:
351 conditions across Qwen3-VL-8B, Qwen3.5-9B and Gemini 2.5 Flash Lite.

## For Thesis Writing

| Need | Open |
| --- | --- |
| Methods, intended claims and limitations | [THESIS_GUIDE.md](THESIS_GUIDE.md) |
| Current tables, figures and completion audit | [reports/matched_gold_v2](reports/matched_gold_v2/README.md) |
| Recommended score table | [thesis_scores.tsv](reports/matched_gold_v2/thesis_scores.tsv) |
| Exact evidence paths for all 351 conditions | [condition_catalog.tsv](reports/matched_gold_v2/condition_catalog.tsv) |
| Folder and version explanations | [Repository structure](docs/REPOSITORY_STRUCTURE.md) |
| Retained supplementary diagnostics and masked inference | [Additional-study handoff](experiments/additional_v1/EXPERIMENT_HANDOFF.md) |

Coverage, scoring and planned analyses are complete. This does **not** mean all
outputs are valid translations or grammar consistently improves performance.
The primary Lezgi sensitivity tables are descriptive. Separate post-hoc Qwen-only
84/83-row cluster-bootstrap tests are in the additional study; do not attach
87-row significance tests to filtered scores. Read the thesis guide first.

## Which Version?

| Family | Status |
| --- | --- |
| **matched_gold_v2** | **Current primary study:** 21 nonempty gold-glossed training supports |
| matched_v1 | Historical matched study: support gloss fields were empty |
| curated_v1, chain_gloss_v2, baseline, legacy, smoke | Earlier material, repair, baseline or pilot experiments |
| experiments/mtob | Separate Ge/Gs/Gl supplementary study, not part of the 351 conditions |
| experiments/additional_v1 | Retained supplement: 37 Qwen3 conditions plus historical Qwen analyses; new Qwen3.5 generation excluded |

Version numbers are local to an experiment family. In particular, chain_gloss_v2
is not matched_gold_v2. XCOMET-XXL is an additional evaluator of the same current
predictions, not another study version. Historical outputs remain available but
must not be pooled with the current study. Luna is not in the primary matrix.

## Where Are the Files?

| Folder | Purpose |
| --- | --- |
| [reports/](reports/README.md) | Reader-facing tables, figures and audits |
| configs/ | Recorded settings, inputs and fingerprints |
| [results/](results/README.md) | Saved model responses |
| [metrics/](metrics/README.md) | Per-condition scores, one real location per collection |
| docs/ | Protocol details, statistical analyses and provenance |
| inputs/ and materials/ | Original PDFs and derived grammar contexts |
| runners/ and scripts/ | Execution, scoring, submission and generation utilities |
| tests/ | Verification tests |
| experiments/mtob/ | Self-contained supplementary experiments and reports |
| archive/ | Superseded or unclassified historical artifacts |

Follow the same family identifier across configs/, results/, metrics/ and docs/.
Current metrics are grouped under metrics/matched_gold_v2/ in two subdirectories:
lexical_and_xcomet_xl/ and xcomet_xxl/. There are no extra metric-view folders or
duplicate aliases. Frozen configurations keep their recorded path strings;
active tools resolve them through a hash-documented path-only migration.
The [structure guide](docs/REPOSITORY_STRUCTURE.md) explains the older families.

## Verification and Reproduction

From this directory, in the recorded local environment:

```bash
venv/bin/python scripts/finalize_gold_v2.py
venv/bin/python scripts/render_gold_v2_appendix.py
venv/bin/python -m unittest discover -s tests -p test_final_gold_v2.py -v
venv/bin/python -m unittest discover -s tests -p test_repository_layout.py -v
venv/bin/python -m unittest discover -s tests -p test_artifact_layout.py -v
```

These commands audit and rebuild the current report. They do not submit jobs,
load generation models or call paid APIs. See the
[corrected protocol](docs/matched_gold_v2/README.md) for execution details.
Do not regenerate frozen configurations or rerun experiments merely to browse
the published results.

Reproduction requires the external sibling Database/2023glossingST/data dataset,
the recorded dependencies and access to the model weights. The repository is not
a preconfigured cluster environment. Local venv/, caches, credentials and Slurm
logs are not publication data. No key is required to read the reports.

GPU jobs use one H100 per job; independent jobs may run in parallel. Production
Qwen precision, prompts, decoding and image inputs are documented in the thesis
guide. Historical settings must not be mistaken for the corrected study's settings.
