# Prediction Collections

- **matched_gold_v2/** contains the corrected primary study: 351 conditions with
  21 nonempty gold-glossed training supports. See the
  [evaluation report](../grammar_context_evaluation/README.md).
- **matched_v1/** preserves the earlier matched-study outputs whose support
  gloss fields were empty. They have not been deleted. They are not valid
  replacements for the corrected primary-study predictions.
- Other family directories contain earlier experiments, diagnostics or pilot
  tests. They must not be pooled into the primary 351-condition matrix.

The historical MTOB/GRAMMAMT comparison uses matched_v1, not matched_gold_v2.
Retaining historical predictions preserves that analysis's provenance; it does
not endorse their use as corrected gold-support results. Neither predictions
nor their hashes were modified for the readable metric-directory views.
