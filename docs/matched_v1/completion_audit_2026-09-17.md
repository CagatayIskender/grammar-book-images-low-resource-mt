# Matched Matrix Completion Audit

Scope: Qwen3, Qwen3.5 and Gemini 2.5 Flash Lite only. Luna is excluded from this assessment.

## Completion

| Model | Complete conditions | Expected / actual records | Empty predictions | Verified XCOMET files |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 | 117/117 | 15999/15999 | 29 | 117/117 |
| Qwen3.5 | 117/117 | 15999/15999 | 12 | 117/117 |
| Gemini 2.5 Flash Lite | 117/117 | 8733/8733 | 268 | 117/117 |

Each model has 12 matched baseline conditions and 105 grammar-context conditions.
There are 351 complete conditions, including 315 context-versus-baseline contrasts.
Record totals count sentence-condition evaluations, not unique test sentences.
Empty predictions remain in the denominator; completion does not mean every sentence
was translated successfully or that translations are semantically correct.

## Verified Checks

- All configuration fingerprints and frozen code/material input hashes pass.
- All results have the expected unique indices and matching source/reference strings.
- All metric files match current result hashes, fingerprints and record counts.
- BLEU and chrF++ were independently recomputed from all 351 prediction files and
  match stored scores within absolute tolerance 1e-8.
- All 351 XCOMET scores are finite and have matching result hash/record-count
  provenance for Unbabel/XCOMET-XL (not XXL).
- All 30254 newly generated row-level prompt hashes match reconstructed fixed prompts;
  reparsing their attempts reproduces the stored translation blocks.
- Within model/language/method, baseline and contexts have identical test ordering,
  21 support examples, decoding policy and, for ModelGloss, predicted input glosses.
- The native and common99 analysis input hashes match current files. Neither analysis
  excludes any condition from these three models. Each contains 630 relevant
  metric comparisons: 315 contrasts times BLEU and chrF++.
- Cross-model source/reference cohorts agree when Qwen Tsez is restricted to first99.
- XCOMET job 5790448 completed in 02:31:04; analysis jobs 5790449 and 5790450
  completed in 01:38:27 and 01:31:33, respectively. All exit codes are 0:0.
- Refreshed matrix/status artifacts report no active jobs or uncovered Qwen conditions.

## Required Interpretation Limits

- Use matched_v1 baselines, not historical baseline scores, for these ablations.
- This is a fixed adaptation of GRAMMAMT strategies, not a verbatim replication of
  the paper's prompts and decoding. The 512-new-token limit is an implementation choice.
- Model-internal comparisons estimate the grammar-context package effect, including
  its context instruction. Gloss-shot baselines already contain glossed examples.
- Native Tsez has 445 rows for each Qwen model but 99 for Gemini. Use analysis_common99
  for descriptive cross-model comparisons; first99 is not a random sample.
- Local and API decoding policies differ. Cross-model differences are configured-system
  comparisons, not isolated architecture effects; no interaction significance is tested.
- TXT/JPG pairs are not all certified content-equivalent. Their contrasts cannot all
  be described as a pure visual-versus-text representation effect.
- Historical API imports have weaker per-request prompt/finish-reason provenance.
  Generated-prompt verification above does not claim to restore that missing evidence.
- Empty predictions: Qwen3 29/15999 (0.18%), Qwen3.5 12/15999 (0.08%), Gemini
  268/8733 (3.07%). No failed rows were dropped or repaired through reprompting.
- Existing analysis keeps its predeclared four-model Holm correction family
  (840 planned baseline metric tests), even when Luna is omitted from reporting.
  This is conservative for the three-model subset; corrected p-values were not
  silently recalculated after observing results.
- In the existing correction family, the only positive significant metric contrast
  among these models is Qwen3 Natugu ModelGloss cheat-sheet TXT, chrF++ delta about
  +2.98, native adjusted p about 0.01672. Completion does not imply general improvement.
- No human adequacy assessment was performed by this audit. Lack of statistical
  significance is not evidence of equivalence or proof that grammar never helps.

Readiness: complete and suitable for the stated matched within-model analyses,
with the interpretation limits above. No further generation job is needed merely
to fill this three-model matrix.
