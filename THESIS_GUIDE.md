# Final Thesis Guide

Verified handover date: **17 September 2026**. Primary experiment family:
**matched_v1**. Primary models: **Qwen3, Qwen3.5 and Gemini 2.5 Flash Lite**.
Luna is excluded from primary thesis comparisons; its historical files are retained.

## Executive Status

The primary matrix is complete. No additional generation job is needed to fill it.
Results, basic metrics, XCOMET-XL and paired statistical analyses have been verified.
This is an implementation/data-integrity assessment, not a guarantee of human
translation adequacy or academic acceptance of every methodological choice.

| Model | Conditions | Expected / recorded evaluations | Empty predictions | XCOMET files |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 | 117/117 | 15999/15999 | 29 (0.18%) | 117/117 |
| Qwen3.5 | 117/117 | 15999/15999 | 12 (0.08%) | 117/117 |
| Gemini 2.5 Flash Lite | 117/117 | 8733/8733 | 268 (3.07%) | 117/117 |

There are **351 conditions: 36 baselines and 315 grammar-context conditions**.
The 40731 evaluations count sentence-condition pairs, not unique sentences.
Empty predictions are present records and remain in the scoring denominator.

## Start Here

- [Final completion and comparability audit](docs/matched_v1/completion_audit_2026-09-17.md)
- [Primary matrix: overview JPG](docs/matched_v1/figures/experiment_matrix_overview.jpg)
- [Primary matrix: detailed JPG](docs/matched_v1/figures/experiment_matrix_detailed.jpg)
- [Overview PDF](docs/matched_v1/figures/experiment_matrix_overview.pdf)
- [Detailed PDF](docs/matched_v1/figures/experiment_matrix_detailed.pdf)
- [Timestamped condition and queue status](docs/matched_v1/CURRENT_STATUS.md)
- [Condition catalog, counts and file paths](docs/matched_v1/experiment_catalog.tsv)
- [Full technical protocol](docs/matched_v1/README.md)
- [English handover and preservation checks](docs/matched_v1/english_handover_check.md)
- [Native-cohort statistical report](docs/matched_v1/analysis_native/report.md)
- [Common99 statistical report](docs/matched_v1/analysis_common99/report.md)

The figures show only the three primary models. The full catalog and original
analysis artifacts retain Luna for provenance; filter their model column for
primary thesis tables. Luna's incomplete rows are not gaps in the three-model matrix.

## Research Question and Defensible Claim

Suggested framing:

> Under a fixed adaptation of GRAMMAMT's gloss-based prompting strategies, this
> study measures the incremental effect of grammar-book context on translation,
> separately by model, prompting method and context material.

The primary contrast is **context minus no-book baseline for the same model,
method and test sentences**. The context and its accompanying instruction are
added together. This estimates that context package, not an instruction-independent
effect of linguistic information.

Cross-model comparisons are secondary and descriptive. The study does not test
model-by-material interactions or isolate architecture from decoding and provider
differences. A significant effect in one model and a nonsignificant effect in
another is not itself a significant difference between models.

## Matrix and Data

Each model has 12 baselines (4 languages x 3 methods) and 105 context conditions
(35 material/source/variant combinations x 3 methods).

| Language / source | Qwen test rows | Gemini test rows | Selected PDF images |
| --- | ---: | ---: | ---: |
| Gitksan / Brown PDF1 | 37 | 37 | 9 |
| Gitksan / Rigsby PDF2 | 37 | 37 | 6 |
| Lezgi / grammar | 87 | 87 | 9 |
| Natugu / grammar | 99 | 99 | 9 |
| Tsez / grammar | 445 | 99 | 9 |

- Every original source has six contexts: cheat sheet TXT/JPG, summary tables
  TXT/JPG, summary text TXT and selected PDF-page JPGs.
- Lezgi adds five Cyrillic contexts; no separate Cyrillic PDF-page condition.
- Gitksan PDF1/PDF2 share test data and method-specific baselines. They are
  alternative grammar sources, not independent languages or test sets.
- The same first 21 glossed support examples are used within each language.
- Lezgi's 87 rows contain 85 distinct sources and 86 distinct source/reference
  pairs. Row count is not the number of independent linguistic items.
- Tsez API evaluation uses the first 99 records, not a random sample.

Use analysis_native for within-model results at each model's full configured
sample size. Use analysis_common99 for cross-model Tsez comparisons: the same
first 99 records for all models. These analyses overlap and are not independent
replications. Other languages already share sample sizes.

## Methods and Baselines

| Method | No-book baseline | Context condition | Interpretation |
| --- | --- | --- | --- |
| Gloss-shot (shot) | Glossed support examples, then translation | Same setup plus book context | Additional book-context effect, not all gloss information |
| Chain-gloss | Explicit gloss followed by translation | Same task plus book context | Gloss and translation saved separately; only translation scored |
| ModelGloss | Fixed externally predicted input gloss | Same predicted gloss plus book context | Book-context effect conditional on the same external gloss |

ModelGloss does not use the target test example's gold gloss as input.
External gloss errors can affect both baseline and context results; gloss quality
is not independently reassessed here.

Use **only matched_v1 baseline/material pairs** for primary ablations.
Historical baseline, legacy, curated_v1, chain_gloss_v2 and MTOB results must not
be pooled with them without a separate comparability argument. Old Shot/Chain
outputs produced with the same prompt are not independent Chain evidence.

## Relationship to Original GRAMMAMT

This is an **adaptation**, not a verbatim replication of Appendix L or its scores.
Present original and implemented prompt templates in the thesis appendix.

- System instructions and output delimiters were adapted. ModelGloss omits the
  paper's warning that a predicted gloss may be wrong. These are methodological
  choices, not necessarily unavoidable technical changes.
- The paper's gloss-free few-shot baseline is not our gloss-shot baseline.
- Local models and decoding differ from the original setup. Do not claim to
  reproduce the paper's greedy decoding or untouched model defaults.
- The **512-new-token cap is our implementation setting**, not a limit established
  from the paper. It limits output, not grammar input. Chain-gloss shares this
  budget between gloss and translation, so methods can be constrained differently.
- We score with **XCOMET-XL**, not the paper's XCOMET-XXL. Do not combine their
  values in an undifferentiated metric column.

## Frozen Generation and Failure Policy

| Setting | Qwen3 / Qwen3.5 | Gemini |
| --- | --- | --- |
| Decoding | Sampling; temperature 0.7, top-p 0.8, top-k 20, min-p 0 | Temperature 0 |
| Penalties | Presence 1.5; repetition 1.0 | As recorded in frozen API configuration |
| Seed | 20260910 + 2 x original test index | 42 |
| Maximum new tokens | 512 | 512 |
| Thinking | Disabled | Reasoning effort none; violations rejected |
| Resources | FP32; one H100 per job; no offload or quantization | OpenRouter, recorded Google provider |

Settings are fixed within each model/method/material contrast, not identical
across local/API backends. A fixed seed does not guarantee identical execution
across hardware, libraries or hidden provider revisions. There is no multi-seed study.

One semantic attempt is retained per sentence. There are no response-conditioned
prompt repairs, best-guess retries or changes to rescue low-quality answers.
Identical transport requests may be retried after API/network failures. A
recoverable translation can be parsed without changing the prompt. Empty,
truncated, refused or unextractable responses remain failures in the denominator.
A malformed/missing gloss does not erase an extractable translation; report
gloss compliance separately. Explicit task glossing is not hidden model reasoning.

Compatible historical first attempts are copied with provenance; later prompt-repair
attempts are not selected as improved predictions. Some historical API records lack
per-request prompt hashes and finish reasons. Reconstructed settings are weaker
evidence than original request logs; do not imply perfect historical reconstruction.

## Statistical Significance: Completed

**Yes, the tests have been run, not merely scheduled.**

| Analysis | Job | Outcome | Runtime |
| --- | --- | --- | --- |
| Native-cohort paired tests | 5790449 | Completed, exit 0:0 | 01:38:27 |
| Common99 paired tests | 5790450 | Completed, exit 0:0 | 01:31:33 |
| XCOMET-XL scoring | 5790448 | Completed, exit 0:0 | 02:31:04 |

SacreBLEU paired bootstrap uses **100000 resamples**, seed **20260915**, BLEU and
chrF++ (word_order=2), followed by **Holm multiple-test correction**. Each cohort
contains all **630 primary metric comparisons**: 315 baseline contrasts x 2 metrics.
TXT/JPG paired tests are saved separately with their own correction family.
XCOMET significance was not tested.

The saved analyses retain the original four-model planned family: **840 planned
baseline metric tests**, including Luna. Missing planned tests are conservatively
treated as p=1. Omitting Luna from reporting does not change these adjusted p-values.
Disclose the preserved family rather than silently choosing a smaller correction
after seeing results. Material-pair correction counts are also in results.json.

All paths below are relative to docs/matched_v1:

| Artifact | Purpose |
| --- | --- |
| analysis_native/scores.tsv | Complete scores at native sample sizes |
| analysis_native/comparisons.tsv | Context-minus-baseline delta, raw and adjusted p-values |
| analysis_native/material_pairs.tsv | JPG-minus-TXT comparisons |
| analysis_common99/scores.tsv | Descriptive cross-model scores on common cohorts |
| analysis_common99/comparisons.tsv | Same-model contrasts on common cohorts |
| analysis_common99/material_pairs.tsv | Common-cohort JPG-minus-TXT comparisons |
| Both results.json files | Input hashes, seeds, signatures, exclusions and correction scope |

Per-condition metrics under metrics/matched_v1 include XCOMET and prediction-hash
provenance. The TSV field chrf means chrF++ with word_order=2, not plain chrF.
Some significance rows label this metric chrF2++.

### Main Statistical Finding

Under the preserved Holm correction, the sole positive significant primary metric
contrast is **Qwen3 / Natugu / ModelGloss / cheat sheet TXT**, approximately
**+2.98 chrF++**, adjusted p approximately **0.01672** in the native analysis.
The common99 result is the same contrast with adjusted p approximately 0.01676,
not independent confirmation. Most contrasts do not establish positive improvement
after correction. Do not generalize one result to every material, language or model.
Report negative and nonsignificant effects as well.

Confidence intervals describe **individual system scores**, not paired score
differences. Sentence bootstrap does not measure generation-seed/API-version
uncertainty. Nonsignificance is not equivalence; a small p-value is not automatically
a practically meaningful improvement. No minimum practically important difference
or human adequacy threshold was established.

## Comparability Audit: Which File to Use

**Keep comparability evidence in the handover.** Matching conditions is necessary
for interpreting effects; significance testing cannot repair unmatched baselines.
A thesis chapter named "Comparability Audit" is optional, but its checks and
limitations belong in Methods, Limitations or an appendix.

- **Final evidence:** [completion audit](docs/matched_v1/completion_audit_2026-09-17.md).
  It verifies 351 conditions, source/reference alignment, input/protocol hashes,
  metric freshness, XCOMET provenance and statistical-input hashes. BLEU/chrF++
  were recomputed from every prediction file. All 30254 newly generated prompt
  hashes match the frozen prompt builder.
- **Historical evidence:** [earlier audit](docs/comparability_audit/README.md).
  Its 15 September findings explain the repairs. Missing results described there
  are not the final status. Preserve it as history, not the final verdict.
- No new audit job is needed merely to deliver this completed matrix. Changes to
  predictions, prompts or scoring would require another appropriately scoped check.

## Required Limitations and Reporting Rules

| Issue | Required disclosure |
| --- | --- |
| Development history | Earlier trials on these data influenced settings. Do not claim an untouched independent development/validation set. |
| Small datasets | Gitksan, Lezgi and Natugu have 37, 87 and 99 rows. Power and generalization are limited. |
| Duplicate Lezgi rows | Repeated linguistic examples are not independent. There is no cluster-bootstrap correction. |
| Support examples | Fixed first 21 examples; no support-selection robustness study. |
| Book/test overlap | No independent leakage-free guarantee for every book. Avoid claims of entirely unseen examples or information. |
| TXT/JPG content | Not all pairs are certified content-equivalent. Content, density and layout can affect differences, not just modality. |
| Source variants | Label Gitksan grammar sources and Lezgi Cyrillic separately. |
| Context instruction | The added material and its instruction form one intervention; their effects are not isolated. |
| ModelGloss | Cached gloss quality can affect outcomes; the same gloss is used on both sides. |
| Output failures | Report empty and gloss-compliance rates. Never drop failed sentences to improve averages. |
| Output cap | Chain-gloss may be constrained more by the same 512-token limit. Report truncation, not missing-data exclusion. |
| Native/common99 | Different purposes, overlapping data, not independent replications. |
| Cross-model inference | Different decoding/provider settings; no causal architecture or interaction claim. |
| Metrics | Automatic proxies, not human adequacy evaluation. |
| XCOMET | XL, not XXL. Target low-resource language coverage is not established; do not use it as the sole arbiter. |
| Multiple comparisons | Report raw and adjusted p-values with the preserved planned family; do not cherry-pick. |
| Reuse provenance | Incomplete historical API request evidence; distinguish reconstruction from recorded verification. |
| Resources | FP32 and one H100 per local job; independent jobs can run concurrently. Not a single-GPU campaign-wide claim. |
| Historical families | Earlier audits/exploratory results are not the final matched comparison. MTOB Ge/Gs/Gl is a separate suite. |
| Luna exclusion | Outside primary scope; budget-limited partial results are not primary evidence. |

## Handover and Reproducibility

Start with this guide, the final audit, English matrix PDFs, the catalog and both
statistical output folders. Use catalog paths to inspect raw results, generated
glosses, failures and per-condition scores. Historical documents retain their dates
and evidence rather than being rewritten as final results.

English documentation does not translate or rename frozen grammar forms, PDFs,
prompts, references, predictions or provenance paths. Keep credentials, environment
files, API keys, caches, virtual environments and personal shell configuration out
of any shareable package. No GitHub push/publication is part of this documentation
update. Reading the results requires no new generation or API charges.

Refresh the catalog and English figures without submitting jobs:

```bash
venv/bin/python scripts/status_matched.py --validate
venv/bin/python scripts/build_thesis_overview.py
```

Outside Slurm, the second command accepts --without-slurm and makes no live queue
claim. Optional statistical verification commands (not unfinished work):

```bash
venv/bin/python runners/matched/analyze.py --samples 100000 --models qwen3 qwen35 gemini25flashlite gpt56luna --cohort native
venv/bin/python runners/matched/analyze.py --samples 100000 --models qwen3 qwen35 gemini25flashlite gpt56luna --cohort common99
```

These overwrite their derived analysis folders and take substantial CPU time.
Including Luna reproduces the saved correction family; filter its rows from primary
reporting. Do not change the model list and describe the result as the same planned
family. This protocol was fixed for the campaign but is not claimed to have been
externally preregistered.

## Separate MTOB Supplement

The [MTOB closure protocol](experiments/mtob/evaluation_v1/README.md),
[live closure report](experiments/mtob/evaluation_v1/CLOSURE_REPORT.md), and
[30-condition matrix](experiments/mtob/evaluation_v1/condition_matrix.tsv)
cover the existing Qwen3/Qwen3.5 Ge/Gs/Gl experiments only. This supplement is
not included in the 351-condition primary matched_v1 matrix above.

Use the closure report's current completion state before citing new MTOB scores
or significance. Original outputs and historical metrics are preserved. New
evaluation retains failures in the denominator and provides both raw and
deterministically extracted translation views. It uses 120 planned paired
BLEU/chrF tests, 100,000 resamples, and one Holm correction family across views.

The original 30-condition MTOB closure has no matched grammar-free baseline: its
conclusions concern Ge/Gs/Gl differences, not a benefit from adding grammar. The
planned source-specific [baseline extension](experiments/mtob/baseline_v1/README.md)
was cancelled at the researcher's request. Its preparation files are not evidence
of completed experiments and must not be cited as baseline results.

The separate [cross-protocol comparison](experiments/mtob/grammamt_comparison_v1/README.md)
uses existing matched_v1 shot, chain_gloss and modelgloss baseline predictions.
It pairs identical sentences/references and recomputes both setups with one
evaluator. Consult its [live report](experiments/mtob/grammamt_comparison_v1/COMPARISON_REPORT.md)
before citing completion or significance. Ninety comparisons, two views and two
metrics define a separate 360-test Holm family. Neither the primary 351-condition
matrix nor the original 120-test MTOB closure family is changed.

Completion verified: Slurm job 5795449 finished successfully in 01:17:38.
All 360 cross-protocol tests completed. Holm correction identifies 217 significant
differences, all favouring the GRAMMAMT baseline configuration: raw BLEU 39,
raw chrF 77, extracted BLEU 39 and extracted chrF 62. These are metric-test
counts across correlated comparisons/views, not 217 independent experiments.
All 90 comparisons have lower MTOB BLEU and chrF in both views; not every
difference is statistically significant. Original input hashes remain verified.

These are valid **cross-protocol translation-performance comparisons**, not
controlled grammar-addition ablations. MTOB uses temperature 0.05 and a 256-token
output cap; matched_v1 uses temperature 0.7 and a 512-token cap. Prompts, examples,
gloss information, seeds and retry policies also differ. Common scoring removes
metric-convention differences, not these experimental confounds. Gitksan PDF1
and PDF2 share their language baseline/test cohort, so they are not independent
datasets. Report absolute differences first; relative changes are undefined at
zero baseline and unstable near zero. Do not mix historical metrics directly.

Actual Gl lengths vary substantially and are not uniformly 100K tokens. Four
Qwen3.5 records were rerun; earlier-round responses are not retained in their
final JSONL. Missing finish metadata means truncation is unknown. Read the
closure report for extraction ambiguities, local ROUGE limitations, CharacTER
definition and historical-denominator effects. Neither a non-significant result
nor two equally low scores establishes equivalence.

## References

- [GRAMMAMT paper, including Appendix L](https://aclanthology.org/2025.acl-long.1447.pdf)
- [SacreBLEU paired significance tests](https://github.com/mjpost/sacrebleu#paired-significance-tests)
- [XCOMET-XL model card](https://huggingface.co/Unbabel/XCOMET-XL)
- [XCOMET-XXL model card](https://huggingface.co/Unbabel/XCOMET-XXL)
