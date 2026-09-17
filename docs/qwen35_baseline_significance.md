# Qwen3.5 Baseline Controls

For each of Gitksan (37), Lezgi (87), Natugu (99), and Tsez (445), run:

- Shot / greedy and sampled_v1.
- ModelGloss / greedy and sampled_v1.
- Explicit Chain / greedy and sampled_v1.

This is 24 conditions in four sequential language jobs, not 24 simultaneous
model allocations. Each job requests exactly one H100 and uses FP32 with no
offload, quantization, or thinking. No grammar text or image is provided. The
same first 21 covered training examples and, for ModelGloss, the same predicted
GlossLM gloss are used. Each condition first checks three longest sources;
full generation is skipped if that check fails. Raw attempts and errors remain
auditable. Generation and XCOMET are separate.

The new baseline prompt is derived from the matching context prompt by removing
the grammar block and grammar-reference instructions. The explicit Chain
retains the gloss-first/translation-second instructions. Greedy uses the audited
runner; sampled_v1 uses its frozen sampled implementation (temperature .7,
top-p .8, top-k 20, generated-token presence penalty 1.5, 512-token cap, two
attempts, source-index-based seed). These are no-grammar controls, not zero-shot
controls. No test reference or gold test gloss enters the prompt.

Outputs: `results/baseline/qwen35/<language>/grammar/original/<condition>_<policy>.jsonl`.
Metrics: matching paths under `metrics/`.
Do not score `.preflight.jsonl` as full-test predictions.

Analysis command, once full baseline outputs exist:

```bash
venv/bin/python scripts/analyze_baseline_significance.py --models qwen3 qwen35 --samples 10000 --output docs/statistical_significance/with_qwen35
```

Qwen3.5 sampled context outputs are paired only with sampled Qwen3.5 baselines;
old greedy outputs are paired with greedy Qwen3.5 baselines. Historical Qwen3.5
chain blocks that used the Shot prompt are excluded as separate Chain evidence.
Missing/incomplete baselines and context files are listed as exclusions, not
silently substituted with Qwen3 or evaluated on overlapping subsets. Full files
with recorded errors retain their empty predictions in the test denominator.

The statistical analysis uses sacreBLEU paired bootstrap (10,000 resamples),
BLEU and chrF++, deterministic seed, per-system confidence intervals, signed
score differences and raw/Holm-adjusted p-values. Correction families span
both metrics and all comparisons within the declared methodological scope.
The historical-control scope does not establish equal precision or prompts in
old runs. The sampled scope matches decoding, but statistical significance still
does not establish practical usefulness or model-independent generalization.

Qwen3 interim result (2026-09-13): 326 metric tests, 21 positive differences with
raw p < .05, zero positive differences significant after Holm adjustment. One
negative difference remains significant. This includes legacy and curated
conditions, not just one prespecified contrast; multiplicity correction is
conservative. Existing results were not edited. See the full TSV, source hashes,
quality flags and exclusions in `docs/statistical_significance/2026-09-13/`.
