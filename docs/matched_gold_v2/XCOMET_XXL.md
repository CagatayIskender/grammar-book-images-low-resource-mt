# XCOMET-XXL Evaluation

The user selected the GRAMMAMT evaluator, `Unbabel/XCOMET-XXL`, for thesis
reporting. The separate scorer was submitted as job **5798139** on 19 September
2026, with dependency `afterok:5798036`. It **completed successfully in 05:57:03**.
All 351 conditions were scored; subsequent XXL analysis 5798146 also completed.
Do not label XL scores as XXL. See `xcomet_xxl_submission.json` for the historical
submission receipt and [the final audit](../../reports/matched_gold_v2/COMPLETION_AUDIT.md) for completion.

The new scorer reads the frozen `matched_gold_v2` predictions and validates all
351 conditions, including actual prompt evidence, before loading any model. It
does not generate translations, call an API, edit configurations, or replace old
metrics. It requires one H100, uses batch size 1 and FP32, and has no quantization,
offload, or second-GPU fallback. Successful production verified GPU fit.

Outputs: `metrics/matched_gold_v2/xcomet_xxl/`, preserving the existing model and
language subpaths. Artifacts explicitly contain `xcomet_xxl`, segment scores,
prediction/configuration/scorer/checkpoint hashes, COMET version, and scale.
Native scores are retained; multiply by 100 only for a clearly labelled table
using the paper's percentage-style display. Empty predictions remain included.
Existing XL metrics are untouched. The frozen gold-v2 status tool checks XL only;
the new scorer independently verifies XXL artifacts before reusing them.

Local validation, without network/model loading:

```bash
venv/bin/python runners/scoring/score_gold_v2_xcomet_xxl.py --dry-run
venv/bin/python -m unittest discover -s tests -p test_gold_v2_xcomet_xxl.py -v
bash -n scripts/scoring/run_matched_gold_v2_xcomet_xxl.sh
```

The job has already been submitted. Do not submit a duplicate. For a future
explicit rerun after checking its state and the completeness of inputs:

```bash
sbatch scripts/scoring/run_matched_gold_v2_xcomet_xxl.sh
```

The script requests 14 hours and 144 GB host RAM; these are conservative initial
limits, not measured XXL runtime requirements. It requests exactly one GPU.
Model access and inference succeeded in the completed production run.
The current environment has unbabel-comet 2.2.7, torch 2.7.1+cu118, and
transformers 4.57.6; versions alone are not evidence of correctness.

The prepared scorer currently targets `matched_gold_v2`, not a hypothetical
future exact-prompt family. If generation protocol changes, its validation and
input family must be explicitly adapted before submitting the scoring job.

Paper: https://aclanthology.org/2025.acl-long.1447.pdf (Section 4.4, Appendix B).
Model card: https://huggingface.co/Unbabel/XCOMET-XXL.
The card warns about uncovered languages, so XXL is not a substitute for BLEU,
chrF++, or translation error analysis. The existing BLEU/chrF++ analysis does not
test XXL. The separate CPU-only XXL analysis **5798146** completed after
successful XXL scoring job 5798139. See [XXL statistics](XXL_STATISTICS.md).
