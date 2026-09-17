# MTOB Qwen Closure Report

This is a separate supplementary study, not part of the 351-condition matched_v1 matrix.
No new generation, API calls, prompt changes or historical result/metric replacement.

Audit: PASS; 4230 records across 30 conditions.

## Protocol and limitations

- Only Ge versus Gs versus Gl is tested. There is no grammar-free baseline; no causal claim about adding grammar is supported.
- PDF extraction replaces the paper's LaTeX preprocessing; Gl on both Qwen models extends the original setup.
- Prompts are reconstructed exactly from Appendix C templates and frozen contexts. Ge/Gs each use two 512-token chunks with 256-token overlap (final chunks may be shorter).
- Embedding model weights are not loaded. Frozen retrieval passages/metadata are checked against the original chunks; embedding retrieval ranking is not recomputed.
- Four Qwen3.5 records have rerun_round=1. Prior round responses are not retained in their final files. Available refusal/reasoning retry histories are counted in audit.json.
- Saved thinking controls and temperature are checked; missing historical runtime parameters cannot be certified. Truncation is unknown where finish metadata was not saved.
- Known refusals/empty/reasoning violations remain in both denominators as empty predictions. Ordinary explanations are not newly classified as hidden reasoning.
- Raw means the saved final response with existing basic label cleanup. Extracted uses reference-blind frozen rules; ambiguous boundaries preserve raw text. Multi-sentence translations are not automatically shortened.
- Both views apply paper punctuation cleanup. BLEU is unsmoothed 13a, 0-1; bootstrap BLEU is the same score times 100. Plain chrF is 0-100, not chrF++.
- ROUGE is the existing local macro-F1 implementation with ASCII tokens; rougeLsum equals rougeL, not official summary-level ROUGE. CharacTER uses word shifts and hypothesis-length normalization, capped at 1; it is not standard CER.
- Corpus-overlap and extraction-quality limitations are inherited; this closure audit does not prove absence of test contamination or independently redo linguistic curation.
- A non-significant difference is not evidence of equivalence. Model-to-model significance and ROUGE/CharacTER significance are not tested.

- Zero observed effects receive conservative p=1 before Holm; the unmodified SacreBLEU strict-tail p-value is also retained as p_sacrebleu.

## Actual Gl context lengths

| Source | GPT-2 tokens |
|---|---:|
| gitksan_pdf1 | 9181 |
| gitksan_pdf2 | 13310 |
| natugu | 90307 |
| lezgi | 18935 |
| tsez | 25903 |

## Condition matrix

BLEU uses the 0-1 scale below. Metrics pending until the scoring stage finishes.

| Model | Source | Condition | Records | State | Raw BLEU | Extracted BLEU | Raw chrF | Extracted chrF | Changed | Ambiguous |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| qwen3 | gitksan_pdf1 | ge | 37 | verified metrics | 0.000000 | 0.000000 | 17.390667 | 18.588710 | 5 | 5 |
| qwen3 | gitksan_pdf1 | gs | 37 | verified metrics | 0.000000 | 0.000000 | 17.198315 | 19.652939 | 7 | 2 |
| qwen3 | gitksan_pdf1 | gl | 37 | verified metrics | 0.000000 | 0.000000 | 19.840276 | 20.098177 | 1 | 2 |
| qwen3 | gitksan_pdf2 | ge | 37 | verified metrics | 0.000000 | 0.000000 | 15.799866 | 17.624045 | 8 | 4 |
| qwen3 | gitksan_pdf2 | gs | 37 | verified metrics | 0.002778 | 0.003550 | 14.691532 | 16.633959 | 10 | 2 |
| qwen3 | gitksan_pdf2 | gl | 37 | verified metrics | 0.000000 | 0.000000 | 20.225299 | 20.225299 | 0 | 1 |
| qwen3 | natugu | ge | 99 | verified metrics | 0.004739 | 0.009161 | 14.351307 | 17.400775 | 28 | 3 |
| qwen3 | natugu | gs | 99 | verified metrics | 0.010662 | 0.019424 | 15.623942 | 19.549404 | 36 | 2 |
| qwen3 | natugu | gl | 99 | verified metrics | 0.012076 | 0.013105 | 15.079132 | 15.792540 | 8 | 1 |
| qwen3 | lezgi | ge | 87 | verified metrics | 0.001804 | 0.002545 | 11.840666 | 13.758890 | 19 | 0 |
| qwen3 | lezgi | gs | 87 | verified metrics | 0.001893 | 0.002628 | 11.935304 | 13.834281 | 15 | 0 |
| qwen3 | lezgi | gl | 87 | verified metrics | 0.005256 | 0.005586 | 16.979128 | 17.221239 | 1 | 0 |
| qwen3 | tsez | ge | 445 | verified metrics | 0.000910 | 0.001036 | 14.697318 | 15.460309 | 61 | 20 |
| qwen3 | tsez | gs | 445 | verified metrics | 0.000000 | 0.000000 | 13.384531 | 16.666708 | 259 | 6 |
| qwen3 | tsez | gl | 445 | verified metrics | 0.000000 | 0.000000 | 17.886850 | 17.886850 | 0 | 2 |
| qwen35 | gitksan_pdf1 | ge | 37 | verified metrics | 0.007022 | 0.007785 | 21.727884 | 22.742732 | 2 | 5 |
| qwen35 | gitksan_pdf1 | gs | 37 | verified metrics | 0.005826 | 0.008291 | 19.570443 | 22.592400 | 6 | 9 |
| qwen35 | gitksan_pdf1 | gl | 37 | verified metrics | 0.005831 | 0.008843 | 19.995214 | 23.536954 | 13 | 9 |
| qwen35 | gitksan_pdf2 | ge | 37 | verified metrics | 0.006129 | 0.007654 | 18.578656 | 20.554603 | 7 | 13 |
| qwen35 | gitksan_pdf2 | gs | 37 | verified metrics | 0.003902 | 0.004804 | 17.620503 | 19.594229 | 7 | 13 |
| qwen35 | gitksan_pdf2 | gl | 37 | verified metrics | 0.002846 | 0.003551 | 16.732253 | 18.412133 | 9 | 12 |
| qwen35 | natugu | ge | 99 | verified metrics | 0.006074 | 0.007823 | 15.454145 | 16.830006 | 11 | 11 |
| qwen35 | natugu | gs | 99 | verified metrics | 0.018646 | 0.020683 | 18.569582 | 19.405945 | 4 | 6 |
| qwen35 | natugu | gl | 99 | verified metrics | 0.011480 | 0.023822 | 15.722544 | 19.844996 | 57 | 19 |
| qwen35 | lezgi | ge | 87 | verified metrics | 0.012384 | 0.013904 | 22.188600 | 22.959827 | 2 | 1 |
| qwen35 | lezgi | gs | 87 | verified metrics | 0.019113 | 0.020825 | 23.081662 | 23.629212 | 2 | 1 |
| qwen35 | lezgi | gl | 87 | verified metrics | 0.004186 | 0.008381 | 14.754117 | 20.043608 | 31 | 1 |
| qwen35 | tsez | ge | 445 | verified metrics | 0.000000 | 0.000000 | 18.171724 | 18.430556 | 8 | 16 |
| qwen35 | tsez | gs | 445 | verified metrics | 0.000000 | 0.000000 | 17.934246 | 18.834701 | 28 | 18 |
| qwen35 | tsez | gl | 445 | verified metrics | 0.001069 | 0.001669 | 14.952743 | 17.410204 | 156 | 28 |

## Statistical analysis

State: complete. Completed tests: 120/120.
Holm-significant metric tests: 12/120.
100,000 paired SacreBLEU bootstrap resamples; seed 20260917; one Holm family across both views and metrics.
See analysis/results.json for raw/adjusted p-values, score differences, signatures and uncertainty.

## Audit trail

- provenance.json: original-input SHA256 values, implementation hashes, versions and frozen settings.
- audit.json: identity, context, prompt, retry, status and count checks.
- records/: final original responses, exact character spans, selected response fields and parser rules.
- metrics/: both views and historical-score differences separated into denominator and extraction effects.
- analysis/pairs/: resumable paired tests bound to exact evaluated inputs.
