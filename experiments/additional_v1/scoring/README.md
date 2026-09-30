# Retained XCOMET Scores

XL and XXL each contain all 37 retained Qwen3 conditions. Prediction hashes,
segment IDs/counts, finite scores, means and mask summaries are checked by
`../publication/finalize.py`.

Original scorer jobs returned partial status because new Qwen3.5 generation was
absent, not because retained Qwen3 scores failed. Valid scores are preserved;
failed submission artifacts were removed. Use the publication manifest, not the
original 74-condition plan, as the delivered scope.

Scores use native COMET scale. Empty translations remain included. Lezgi84/83 and
Tsez445/99 are separate views. No XCOMET significance tests were performed.
Scoring code is retained as provenance, not a new submission instruction.
