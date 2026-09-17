# Interim Effect Interpretation

Evidence: `docs/statistical_significance/with_qwen35/results.json`, 474 metric
comparisons from paired sacreBLEU bootstrap with 10,000 resamples. No claims
below rely on the still-missing Lezgi/Natugu/Tsez Qwen3.5 baselines. This is a
snapshot, not the final matrix. No practical-effect threshold was specified.

## Size versus significance

| Comparison | Baseline | Context | Difference | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- |
| Qwen3 Lezgi original cheat-sheet TXT sampled Chain, BLEU | 3.7486 | 7.4969 | +3.7482 | .02460 | .96390 |
| Same comparison, chrF++ | 18.7598 | 21.5979 | +2.8381 | .00570 | .30777 |
| Qwen3.5 Gitksan PDF1 summary TXT sampled Chain, BLEU | 4.7859 | 4.9713 | +.1855 | .32107 | 1.00000 |
| Same comparison, chrF++ | 24.7000 | 25.7310 | +1.0309 | .12889 | 1.00000 |

These example context files have no recorded output errors. They illustrate
effect size, not a selected confirmatory hypothesis. The Qwen3 Lezgi baseline
is greedy and uses the historical prompt; its sampled explicit-Chain context
condition changes more than grammar availability. The substantial numerical
increase therefore does not identify grammar's isolated effect. Doubling a
small BLEU value does not imply doubling translation quality.

For the 12 currently available matched-decoding Qwen3.5 Gitksan comparisons,
the median delta across conditions is -.124 BLEU and -.060 chrF++; ranges are
[-1.298, +.875] and [-3.575, +1.678]. Five BLEU deltas and six chrF++ deltas
are positive. None is significant even before multiplicity correction. One
condition contains a recorded invalid output retained as empty, not dropped.
These summaries are descriptive across correlated conditions, not a pooled
effect estimate or 12 independent replications.

## What the evidence explains

- Improvements are not consistent across languages, materials and metrics.
  Large isolated maxima cannot represent the whole experimental matrix.
- The declared Holm families are broad. A raw p below .05 can disappear after
  correction across many tests. Do not change the correction family after seeing
  results to obtain significance; a targeted new study requires prespecification.
- Test sizes are Gitksan 37, Lezgi 87, Natugu 99, Tsez 445. Small test sets can
  limit precision, but lack of power is a possible contributor, not a demonstrated
  explanation of every nonsignificant result. Tsez effects in the available
  analysis are also numerically small despite its larger test set.
- Model/decoding/prompt differences confound some comparisons. The matched
  Qwen3.5 baseline removes the decoding mismatch for sampled Gitksan conditions;
  it currently supplies no evidence of a consistent grammar-context advantage.
- Recorded failures include truncation and repetitions. Four full-condition
  `invalid_chain_format` records do contain a final translation, but a multiline
  gloss violates the strict two-line parser. Their original files and failure
  labels remain unchanged. Recovering translations would require a separately
  documented, uniformly applied evaluation-parser sensitivity analysis, not a
  claim that the original jobs passed.
- Prior manual inspection found copied source strings in glosses and incorrect
  lexical meanings. This supports concern about using the context correctly;
  it does not prove an attention, vision, or knowledge mechanism.

Plausible but untested explanations include limited lexical coverage in grammar
summaries and difficulty selecting useful information from a long context. No
token-attention analysis, dictionary-coverage audit or randomized matched control
currently establishes those mechanisms. Do not state them as established causes.

## Thesis-safe conclusion

Some conditions show numerical improvements, particularly Qwen3 Lezgi, but the
current evidence does not demonstrate a robust, general statistically significant
improvement over baseline after the declared multiple-testing correction. This
does not establish equivalence or prove that grammar information is useless.
Matched Qwen3.5 evidence is currently limited to Gitksan. Practical value remains
unresolved without a predefined meaningful effect size and translation-quality
assessment beyond automatic scores.

## Continuation

Eight jobs with no outputs are being retried excluding nodes
`lrz-hgx-h100-015,lrz-hgx-h100-026`, along with the three missing Qwen3.5 baseline
suites. Precision, images and prompts are unchanged. The 0:53 failures occurred
on node 015 with no application log; their exact cause/owner is not established.
The node exclusion is an operational mitigation, not a proven software fix.
The eight recorded full-output failures and one failed preflight are not blindly
resubmitted. New statistical output will be versioned separately after the retry
jobs terminate. Existing reports are retained.
