# Cross-Protocol Translation Comparison

Compare existing Qwen3/Qwen3.5 MTOB Ge/Gs/Gl translations against the same model's
existing GRAMMAMT `matched_v1` no-book baselines: shot, chain_gloss, modelgloss.
No new MTOB baseline is generated. No cancelled baseline jobs are restarted.
API models, grammar-augmented matched conditions and the main 351-condition
matrix are outside this analysis.

The 30 MTOB conditions and 24 distinct baseline conditions form 90 paired
comparisons. Gitksan PDF1/PDF2 share their language baseline. Test sizes are
Gitksan 37, Natugu 99, Lezgi 87 and Tsez 445, with exact source/reference matching.

## Reproduction

From this directory, using the existing shared environment:

```bash
../../../venv/bin/python -m unittest -v test_compare
../../../venv/bin/python compare.py audit
../../../venv/bin/python compare.py score
bash submit.sh --dry-run
bash submit.sh
```

`submit.sh` sends only the long significance analysis: one CPU, 24 GB RAM,
two-hour limit, no GPU. Alternatively run `compare.py analyze` on a CPU
allocation. `compare.py report` regenerates the report from verified scores.
All phases are offline and do not import inference backends. Scores and tests
are reusable only when source, parser and scored-input hashes match.

## Frozen Evaluation Rules

- The MTOB closure audit must still pass; original files remain read-only.
- All expected records are retained, including empty/refusal/reasoning failures.
- GRAMMAMT's requested gloss and `FINAL_TRANSLATION:` marker are structural, not
  part of the translation. A chain response without its required translation
  section yields an empty translation. Multiple markers fail the audit.
- Both setups then use the same frozen MTOB response-view parser. Raw means the
  translation response, with required framing removed. Extracted additionally
  removes only explicitly delimited explanations; no first-paragraph shortcut.
- The audit records any difference from stored baseline predictions. In this
  dataset, the adapter must be checked to ensure it does not accidentally score
  generated glosses or select a reference-favoured answer.
- Available truncation metadata is reported, not inferred for MTOB. No answer
  is repaired or regenerated. No unsuccessful sentence is filtered out.
- Common punctuation cleaning and one evaluator produce BLEU, plain chrF,
  local ROUGE and CharacTER. The report documents scales and implementation
  limits. Stored historical metrics are not mixed into the new comparison.
- A separate family contains all 360 planned BLEU/chrF tests: 90 comparisons x
  two views x two metrics; 100,000 bootstrap resamples; seed 20260917; Holm.
  Identical observed scores are not declared significantly different.
- Percentage changes are undefined at zero baseline, and unstable near zero.
  Prefer absolute score differences and adjusted significance when discussing
  effect size. These are not percentages of correctly translated sentences.

## What This Supports

It supports comparisons of translation quality between **complete experimental
configurations** on the same sentences. It does not isolate the effect of adding
grammar: prompts, demonstration/gloss information, output limits, sampling and
retry policies differ. A significant difference is not evidence that one of
those factors alone caused it. Do not describe non-significance as equivalence.

See [COMPARISON_REPORT.md](COMPARISON_REPORT.md) for current completion,
`comparisons.tsv` for all condition-level numbers, and `significance.tsv` for
the complete corrected test family once the analysis finishes.
