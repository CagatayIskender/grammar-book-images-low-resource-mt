# MTOB No-Book Baseline v1

## Status: Cancelled

This proposed extension was cancelled at the researcher's request. The scripts
below are retained for provenance, not as instructions to resume the study.
Do not report these planned conditions as completed baseline experiments.
The completed [cross-protocol comparison](../grammamt_comparison_v1/COMPARISON_REPORT.md)
instead uses existing GRAMMAMT matched_v1 baseline predictions without generating
new translations. The remainder describes the cancelled proposal.

Ten Qwen-only conditions: two models times five source-specific Appendix C
prompts. Source cohorts: Gitksan PDF1 37, Gitksan PDF2 37, Natugu 99, Lezgi 87,
Tsez 445. Total: 1,410 baseline records. Gemini and Luna are not included.
Gitksan source variants have different location wording, so separate baselines
are retained rather than silently sharing one.

Only the grammar-context block is removed. The existing Appendix C template,
language/location, one-user-message format, temperature 0.05, 256-new-token cap,
seed schedule and within-invocation refusal/reasoning retry logic are unchanged.
Thinking is disabled. Generation requests one H100, FP32, no CPU offload and no
quantization. No images are used in this text-only MTOB baseline.

All terminal records, including failures, are preserved on resume. There are no
extra response-conditioned rescue rounds after a completed baseline invocation.
The four historical Qwen3.5 grammar records with rerun_round=1 remain a documented
comparability limitation: their extra rounds/changed seeds cannot be matched
simultaneously by one baseline across all three grammar conditions. Earlier
overwritten answers are unavailable. Do not claim a perfect one-variable ablation
for those records, or that new baseline generation reproduces unrecorded historical
library/device settings exactly. Saved baseline policies and package versions are
recorded under manifests/; failed outputs remain in the scoring denominator.

## Run

From experiments/mtob:

```bash
export PYTHONPATH="$PWD/vendor:$PWD"
../../venv/bin/python prepare_baselines.py
../../venv/bin/python baseline.py list
bash baseline_v1/submit.sh --dry-run
bash baseline_v1/submit.sh
```

Ten independent single-GPU generation jobs run in parallel when resources allow.
Requested limits: Gitksan 30 minutes each; Natugu/Lezgi 1 hour; Tsez Qwen3 2 hours,
Qwen3.5 1 hour. Previous Ge/Gs generation latency totals were approximately
2-5 minutes for Gitksan, 3-10 minutes for Natugu/Lezgi, and 18-55 minutes for Tsez.
These estimates exclude some startup/retry overhead, so the requests include a
buffer; they are limits, not promises of runtime.

One dependent analysis job requests 1 CPU, 24 GB RAM and 2 hours, no GPU. It starts
only after all ten generation jobs succeed. It scores both raw/extracted views
with the frozen evaluation_v1 parser and full denominators, then runs a separate
family of 120 baseline-Ge/Gs/Gl BLEU/chrF tests (100,000 paired bootstrap samples,
seed 20260917, Holm correction). Existing evaluation_v1 artifacts are read-only.

## Read Results

- results/: original baseline JSONL responses and retry history.
- manifests/: frozen generation policies and library versions.
- evaluation/comparisons.json: absolute scores, signed differences and percentage improvements.
- evaluation/significance.json: raw and Holm-adjusted p-values, separate from the old grammar-pair family.
- evaluation/progress.json: completed test count and explicit errors.

For higher-is-better metrics, percentage improvement is
100 * (context - baseline) / baseline. For CharacTER use
100 * (baseline - context) / baseline. A zero baseline produces null, not infinity.
Near-zero denominators can make percentages misleading: always show original
scores and absolute differences. BLEU scores are 0-1 here; significance BLEU
is the equivalent 0-100 scale. Non-significance does not establish equivalence.
This is a supplementary MTOB comparison, not part of the main 351-condition matrix.
