# Baseline Statistical Significance

sacreBLEU 2.6.0; paired bootstrap, 10000 resamples, seed 20260913.
BLEU: 13a, mixed case, exp smoothing. chrF++: character order 6, word order 2.
Holm correction is applied across both metrics and all language/material comparisons within each declared scope.

## Summary

| Scope | Language | Metric | Tests | Significant gains | Significant losses |
| --- | --- | --- | --- | --- | --- |
| historical_same_model | Gitksan | BLEU | 60 | 0 | 0 |
| historical_same_model | Gitksan | chrF2++ | 60 | 0 | 0 |
| historical_same_model | Lezgi | BLEU | 45 | 0 | 0 |
| historical_same_model | Lezgi | chrF2++ | 45 | 0 | 0 |
| historical_same_model | Natugu | BLEU | 35 | 0 | 0 |
| historical_same_model | Natugu | chrF2++ | 35 | 0 | 1 |
| historical_same_model | Tsez | BLEU | 16 | 0 | 0 |
| historical_same_model | Tsez | chrF2++ | 16 | 0 | 0 |
| sampled_decoding_confounded | Gitksan | BLEU | 2 | 0 | 0 |
| sampled_decoding_confounded | Gitksan | chrF2++ | 2 | 0 | 0 |
| sampled_decoding_confounded | Lezgi | BLEU | 1 | 0 | 0 |
| sampled_decoding_confounded | Lezgi | chrF2++ | 1 | 0 | 0 |
| sampled_decoding_confounded | Natugu | BLEU | 1 | 0 | 0 |
| sampled_decoding_confounded | Natugu | chrF2++ | 1 | 0 | 0 |
| sampled_decoding_confounded | Tsez | BLEU | 3 | 0 | 0 |
| sampled_decoding_confounded | Tsez | chrF2++ | 3 | 0 | 0 |

## Interpretation

- Conditional on this sentence test set and single model runs; not training-seed uncertainty.
- Historical baseline prompt language/precision and context prompt designs may differ; not a randomized causal grammar ablation.
- Sampled results versus greedy baselines have a decoding confound.
- Empty predictions retained in full test sets; incomplete files excluded, with selection-bias risk.
- CIs are sacreBLEU per-system score intervals, NOT confidence intervals of the paired delta.
- No minimum practically important delta specified; significance alone does not establish usefulness.
- Sentence-level bootstrap assumes independent units; document grouping is unavailable.
- This is an interim snapshot; later experiments require rerunning tests and Holm adjustment.

Qwen3.5 and API models are not paired with a different-model baseline here.
Complete raw p-values, adjusted p-values, deltas, quality flags, filenames and signatures are in `comparisons.tsv`.
All excluded files and reasons are recorded in `input_manifest.json`.
A positive significant difference is statistical evidence on these outputs, not proof of practical benefit or grammar-only causation.

Source: https://github.com/mjpost/sacrebleu#paired-significance-tests-for-multi-system-evaluation
