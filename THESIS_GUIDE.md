# Thesis Guide

Primary study: **matched_gold_v2**.
Primary models: **Qwen3-VL-8B-Instruct, Qwen3.5-9B and Gemini 2.5 Flash Lite**.
Luna is excluded. This is an adapted GRAMMAMT study, not an exact replication.

## Study Scope and Sources

The primary study includes completed predictions, XL/XXL scoring and planned
statistical analyses. The reporting restrictions below apply to these results.
Complete records do not mean that every model produced a successful translation.

- [Final handover index](reports/matched_gold_v2/README.md)
- [Final completion and comparability audit](reports/matched_gold_v2/COMPLETION_AUDIT.md)
- [Overview matrix](reports/matched_gold_v2/figures/experiment_matrix_overview.jpg)
- [Detailed matrix, PDF](reports/matched_gold_v2/figures/experiment_matrix_detailed.pdf)
- [Print-friendly six-page matrix](reports/matched_gold_v2/figures/experiment_matrix_appendix.pdf)
- [Condition catalog and artifact paths](reports/matched_gold_v2/condition_catalog.tsv)
- [Recommended thesis score table](reports/matched_gold_v2/thesis_scores.tsv)
- [Frozen full-cohort score table](reports/matched_gold_v2/scores.tsv)
- [Statistical output summary](reports/matched_gold_v2/significance_summary.tsv)
- [Lezgi descriptive sensitivity scores](reports/matched_gold_v2/lezgi_sensitivity_scores.tsv)
- [Final input hashes and audit scope](reports/matched_gold_v2/audit.json)
- [Metric collections](metrics/README.md)

Use only matched_gold_v2 for the primary-study findings. Within
`metrics/matched_gold_v2/`, `lexical_and_xcomet_xl/` and `xcomet_xxl/` evaluate
the same predictions with different metrics; they are not separate experiments.

## Research Question and Claim

Suggested framing:

> Under a fixed adaptation of GRAMMAMT's gloss-based prompting strategies, this
> study measures the incremental effect of grammar-book context on translation
> within each model and method, and compares text and image context packages.

The intervention includes both the context and its accompanying instruction.
It is not an instruction-independent estimate of linguistic information.
The completed results do not establish a consistent general improvement.
Report positive, negative and nonsignificant effects, with effect sizes and
adjusted p-values. Nonsignificance does not establish no effect or equivalence.

Cross-model comparisons are secondary and descriptive. No model-by-material
interaction is tested. Significance in one model but not another does not prove
a difference between models. Architecture, decoding and provider effects are not isolated.
Do not change the protocol or select additional trials merely to obtain significance.

## Completed Matrix

There are **351 conditions: 36 no-book baselines and 315 grammar-context conditions**.
Each model has 12 baselines and 105 context conditions.

| Model | Completed conditions | Sentence-condition records |
| --- | ---: | ---: |
| Qwen3 | 117 | 15999 |
| Qwen3.5 | 117 | 15999 |
| Gemini 2.5 Flash Lite | 117 | 8733 |
| Total | 351 | 40731 |

These are repeated evaluations, not 40,731 distinct sentences.

| Language / source | Qwen rows | Gemini rows | Selected PDF images |
| --- | ---: | ---: | ---: |
| Gitksan / Brown PDF1 | 37 | 37 | 9 |
| Gitksan / Rigsby PDF2 | 37 | 37 | 6 |
| Lezgi / grammar | 87 | 87 | 9 |
| Natugu / grammar | 99 | 99 | 9 |
| Tsez / grammar | 445 | 99 | 9 |

Original sources have six contexts: cheat sheet TXT/JPG, summary tables TXT/JPG,
summary text TXT and selected PDF-page JPG. Lezgi has five additional Cyrillic
contexts, without a separate Cyrillic selected-page condition.
Gitksan sources share one language/method baseline; they are not independent test sets.
Markdown sources are retained, not counted as duplicate TXT experiments.

Use native cohorts for within-model full-configured-test analyses.
Use common99 for descriptive cross-model Tsez comparisons: the same first 99,
not a random sample. The two cohorts overlap and are not independent replications.

## Methods and Gold Glosses

| Method | No-book baseline | Context condition |
| --- | --- | --- |
| Gloss-shot, named shot | 21 gold-glossed training supports; translate test source | Same setup plus grammar context |
| Chain-gloss | Same supports; generate test gloss, then translation | Same task plus context |
| ModelGloss | Same supports; externally predicted test gloss | Same predicted gloss plus context |

Only the translation is scored. Chain gloss compliance is reported separately.
No-book does not mean no grammar information: gold support glosses are still present.
No oracle test-gold-gloss condition is included.

Gold training annotations are aligned with the original support sources:
Gitksan uses official SIGMORPHON data_v1 at commit
190689ac81935359c69a46463c48e25e63e601f7; other languages use local uncovered training files.
[Support manifests](docs/matched_gold_v2/support/) retain indices and provenance.
The selections are not claimed to reproduce the paper authors' exact support order.

### Support Selection

All 108 Lezgi conditions use training indices 0-4 and 6-21: 21 supports and
87 test rows. Training index 5 is excluded because its source exactly overlaps
the test set. Indices are zero-based. Other languages use their fixed first
21 training supports. The same support set is used across methods and materials
within each language.
There is no exact support-source/test-source match, but this is not a guarantee
against partial, paraphrastic or book/test overlap.

### Actual Input Evidence

The final audit verifies every saved system/user message against the frozen
prompt builder and each record's support identity. Each support set contains
21 aligned, nonempty gold glosses. ModelGloss uses the cached predicted target
gloss; it does not read the target gold gloss or English reference into the task input.

Tests check that changing the test gold-gloss or reference fields does not
change the task prompt. Saved-message validation is not an independent
reconstruction of Qwen processor-serialized token sequences.

A test string can occur incidentally in training/context text even though its
test field was not injected. Some external predictions exactly match gold;
that alone is not evidence of oracle input. Grammar images are hash-bound but
not independently OCR-audited for test-answer overlap.

## Lezgi Reference-Quality Decision

Three source-dataset references, indices **37, 62 and 81**, are literal **nan**.
They are not valid English references and are distinct from failed model outputs.

- Preserve the frozen **87-row** results and planned statistics as protocol artifacts.
- Use the **84-row valid-reference sensitivity** for descriptive Lezgi performance.
- Also report the **83-row sensitivity**, excluding index **11**, whose reference
  occurs inside the longer translation of training support index **19**.
- Apply identical masks to every model/method/baseline/context, irrespective of scores.
- Retain all model failures among the retained rows. Do not invent references.
- Recompute lexical scores from the same predictions and average matching XL/XXL
  sentence scores. No new translation or COMET inference is needed.
- These are post hoc descriptive sensitivities. **No filtered-cohort p-values
  were computed; 87-row p-values must not be attached to 84/83-row tables.**
- Full-cohort Lezgi significance is reference-contaminated and is not confirmatory
  evidence of grammar improvement. Conclusions about Lezgi should remain descriptive.

The 87-row dataset has 85 distinct sources and 86 distinct source/reference pairs.
Sentence bootstrap is not cluster-bootstrap; repeated linguistic items are not
fully independent. Removing one known overlap does not certify leakage-free evaluation.

## Relationship to GRAMMAMT

This is an **adaptation**, not verbatim Appendix L replication or a reproduction
of published scores. Include the actual prompt templates in the thesis appendix.

- System instructions and output delimiters were adapted.
- ModelGloss omits the paper's warning that a predicted gloss may be wrong.
- The paper's gloss-free few-shot baseline is not this gold-support gloss-shot baseline.
- Models, decoding, seeds and support selection differ.
- The **512-new-token cap is an implementation choice**, not an established paper limit.
- XCOMET-XXL was selected for thesis reporting; XCOMET-XL is a separately labelled supplement.
- Matching the evaluator alone does not make the experiment an exact replication.

These adaptations do not prevent within-model/method matched context comparisons.
They must remain visible, and published GRAMMAMT scores must not be treated as
directly matched controls.

## Frozen Generation and Failure Policy

| Setting | Qwen3 / Qwen3.5 | Gemini |
| --- | --- | --- |
| Decoding | Sampling, temperature 0.7, top-p 0.8, top-k 20, min-p 0 | Temperature 0 |
| Penalties | Presence 1.5, repetition 1.0 | Frozen API configuration |
| Seed | 20260910 + 2 x original test index | 42 |
| Maximum new tokens | 512 | 512 |
| Thinking | Disabled | Reasoning effort none; violations rejected |
| Resources | FP32, one H100 per job, no offload/quantization | OpenRouter, recorded Google provider |

Settings are fixed within each model/method across materials, not across backends.
A fixed seed does not eliminate hardware/provider variability; no multi-seed study exists.
Chain-gloss shares its output budget between gloss and translation.

One persisted semantic attempt per sentence is retained. No response-conditioned
prompt repairs or best-answer selection are used. Transport retries may repeat an
identical API request. The frozen parser uses FINAL_TRANSLATION or a permitted
single-line fallback; it does not guarantee that every recoverable linguistic
answer is extracted. Truncated outputs are treated as empty by this policy.
Empty/refused/unextractable/reasoning-violating outputs remain in denominators.
Missing generated gloss does not erase an otherwise extractable translation.

Saved Gemini responses, including failures, are retained without answer-based
selection. Missing records may involve repeated identical requests. Responses
lost before persistence are unavailable, so the saved dataset is not an
exhaustive record of every API response received.

## Statistical Analyses

All planned outputs are available under docs/matched_gold_v2.

| Directory | Scope |
| --- | --- |
| analysis_native | Full configured cohorts, BLEU and chrF++ |
| analysis_common99 | Shared first-99 Tsez cohort, BLEU and chrF++ |
| analysis_xcomet_xl | XL, both cohorts |
| analysis_xcomet_xxl | XXL, both cohorts |

Each has scores.tsv, comparisons.tsv, material_pairs.tsv, results.json and report.md.
The final handover joins scores without mixing metrics or experimental families.
The recommended thesis_scores.tsv uses 84 valid-reference rows for Lezgi and
explicitly marks those rows as descriptive-only. It does not attach 87-row
p-values to filtered scores. The original scores.tsv remains available separately.

### BLEU and chrF++

SacreBLEU paired bootstrap uses **100000 resamples**, seed **20260915**.
chrF uses word_order=2, so it is **chrF++**, sometimes labelled chrF2++ in outputs.
Each cohort has 630 baseline/context metric tests (315 pairs x 2 metrics), with
one Holm family; its 216 TXT/JPG metric tests form a separate family.
The reported system_ci_half_width is an individual-system interval, **not a
paired-delta interval**. Do not draw it as uncertainty around a difference.
Metric signatures specify case, tokenizer, smoothing, sample count and version.

### XL and XXL

Each uses 100000 two-sided null-centered paired sentence bootstrap resamples,
plus-one p-values and deterministic seeds derived from 20260919.
95% percentile intervals describe paired mean differences and are unadjusted.
Each metric separately pools both cohorts into 630 baseline/context and 216
TXT/JPG tests, with separate Holm families. This is not SacreBLEU's test implementation.

The four metrics/cohorts do not share one universal correction family. Do not
cherry-pick a significant metric and claim overall familywise control.
Overlapping cohorts and repeated language/source tests are not independent confirmations.
Neither bootstrap measures generation-seed/provider uncertainty.
No minimum practically important difference or human adequacy threshold was set.

### Reading the Findings

Use the [statistical summary](reports/matched_gold_v2/significance_summary.tsv)
and the original comparison tables, including
negative results. The full native lexical analysis has one positive adjusted
metric test for Qwen3.5: Tsez Chain-gloss with cheat sheet TXT has
**+1.378 chrF++**, Holm-adjusted **p = 0.00630**, on 445 rows.
Common99 has no positive adjusted baseline test; it is a smaller overlapping
cohort, not an independent replication.
XL and XXL do not establish a positive adjusted baseline effect.
This does not imply every raw score difference is zero or that systems are equivalent.
Lezgi tests retain their original reference-quality caveat regardless of sign.

XXL scores remain on their native scale; multiply by 100 only in explicitly
labelled percentage-style tables. Never compare an XL baseline with an XXL context.
Relative improvements are undefined at zero and unstable near zero; prefer absolute deltas.

## Required Limitations

| Issue | Disclosure |
| --- | --- |
| Development independence | Settings were informed by prior trials; no untouched independent development set is claimed |
| Sample sizes | Small Gitksan/Lezgi/Natugu cohorts limit precision and generalization |
| Support selection | Fixed 21 examples; no support-selection robustness study |
| References and overlap | Lezgi 84/83-row sensitivities; no claim of entirely unseen information |
| TXT/JPG | Not all pairs certified content-equivalent; content/layout may contribute, not just modality |
| Context intervention | Context and its instruction change together |
| ModelGloss | External gloss errors are shared within pairs; gloss quality not independently reassessed |
| Output cap/parser | Failures included; 512-token cap may constrain chain more |
| Cross-model | Different decoding/providers; no causal architecture or interaction claim |
| Automatic metrics | Not human adequacy; XCOMET language coverage not established |
| Duplicates | Lezgi repeated rows not cluster-resampled |
| Multiplicity | Separate, preserved families; no metric cherry-picking |
| API provenance | Responses lost before persistence are unavailable; saved responses retained |
| Resources | One H100 per local job, FP32; parallel independent jobs permitted |
| Study boundaries | Luna and other experiment families are excluded from the primary 351-condition matrix |

Possible explanations for weak gains (context relevance, visual readability,
long-context use, imperfect glosses, output budget) are hypotheses, not established causes.

## Supplementary MTOB Study

The separate [MTOB closure](experiments/mtob/evaluation_v1/CLOSURE_REPORT.md)
covers 30 Qwen Ge/Gs/Gl conditions and 4230 records, not the primary 351 conditions.
It preserves original outputs and evaluates raw/extracted views, with 120 planned
BLEU/chrF tests, 100000 resamples and one Holm family. There is no MTOB-specific
grammar-free baseline. Ge/Gs/Gl contrasts therefore compare grammar-context
strategies rather than the effect of adding grammar to a matched no-book control.

The [MTOB/GRAMMAMT comparison](experiments/mtob/grammamt_comparison_v1/COMPARISON_REPORT.md)
uses **historical matched_v1 baselines with empty support glosses**, not corrected baselines.
Its completed 360-test family is separate; do not relabel it as a gold-v2 comparison.
Common evaluation does not remove prompt/support/decoding/retry confounds.
MTOB uses temperature 0.05 and cap 256 versus matched_v1's 0.7 and cap 512.
Actual Gl lengths are not uniformly 100K tokens. Four Qwen3.5 records have a
prior-attempt provenance gap: only their retained final responses are available.
Missing finish metadata means truncation is unknown.
Read the closure report for extraction ambiguity, ROUGE limits and the CharacTER
definition. MTOB findings must remain separate from the primary study.

## Reproducibility Evidence

The condition catalog links configurations, recorded predictions and metric
files. The completion audit verifies saved prompts, support identities, expected
record counts and prediction-bound XL/XXL scores. Statistical reports identify
their input hashes, cohorts and test settings.

Use the [file guide](reports/matched_gold_v2/FILE_GUIDE.md) for locating evidence
and the repository [README](README.md) for environment requirements and rebuild
commands. Preserve original language forms, references and predictions when
preparing thesis tables. Lezgi's filtered values remain descriptive; a
confirmatory filtered-cohort analysis is outside the reported study.

## References

- [GRAMMAMT, Appendix L](https://aclanthology.org/2025.acl-long.1447.pdf)
- [SacreBLEU significance](https://github.com/mjpost/sacrebleu#paired-significance-tests)
- [XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL)
- [XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL)
