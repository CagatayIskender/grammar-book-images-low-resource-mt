# MTOB Qwen Evaluation v1

Scope: 30 existing Qwen3/Qwen3.5 experiments, five grammar sources, Ge/Gs/Gl,
4,230 records. This is a supplementary study, separate from matched_v1.
No translation generation, API inference, or model weights are used.
Original prompts, results, historical metrics and configurations remain read-only.

## Run

From `experiments/mtob/`:

```bash
export PYTHONPATH="$PWD/vendor:$PWD"
../../venv/bin/python -m unittest discover -s tests -p test_closure.py -v
../../venv/bin/python finalize_results.py --stage audit
../../venv/bin/python finalize_results.py --stage score
../../venv/bin/python finalize_results.py --stage analyze
# Alternatively, run all three stages in order:
../../venv/bin/python finalize_results.py --stage all
../../venv/bin/python list_experiments.py
```

Run long evaluation on CPU compute nodes, not the login node:

```bash
bash submit/evaluation/submit_evaluation_v1.sh --dry-run
bash submit/evaluation/submit_evaluation_v1.sh
```

Two jobs request **1 CPU, 24 GB RAM, 2 hours each, zero GPUs**. Analysis has
an `afterok` dependency on scoring. No production experiment is submitted.
If submission fails after the first job, its ID is preserved in `submissions/`.
Completed condition scores and paired tests are reused only when their input,
parser, implementation and configuration bindings match. Re-run the failed stage
after resolving an error. Do not delete or bypass a failed audit to force scoring.

## Offline operation

The closure CLI denies network connections and filesystem writes outside this
directory, including temporary files. It uses tiktoken's GPT-2 constructor with
local `tokenizer/vocab.bpe` and `tokenizer/encoder.json`, checked against the
checksum constants shipped in tiktoken. These public tokenizer data files were
provisioned once during setup because the current node's temporary cache was absent.
No model or inference endpoint was contacted. Provisioning is not part of the CLI.

## Frozen extraction policy

Rules are in `../mtob_grammar/response_views.py`, version
`mtob-response-views-1.0.0`; the source hash is captured before scoring.
The final saved `retry_response` is preferred, otherwise `initial_response`,
otherwise `prediction`. This priority uses field availability, not score or reference.

- Raw: existing leading plain `English translation:` cleanup only, then paper punctuation cleanup during scoring.
- Extracted: recognize explicit `Translation:`/`English translation:` labels, including Markdown headings/bold.
- Remove a following standalone Explanation, Breakdown, Notes or Translation notes section, including descriptive heading suffixes such as "of the translation".
- Explanation-first responses require a unique explicit translation-labelled section.
- Multiple labels or unclear boundaries preserve raw text and flag ambiguity. Never select the first sentence or paragraph heuristically.
- Known refusal, empty and reasoning-violation records receive blank predictions in both views and remain in the denominator.
- Ordinary explanations are not newly classified as reasoning violations.
- Character spans refer to the saved original response before punctuation cleanup. Unknown finish metadata remains unknown.

The same policy applies to every condition before computing any scores.
This is a post-hoc sensitivity analysis, not an externally preregistered experiment.

## Metrics and inference

BLEU: unsmoothed, four orders, Tokenizer13a, effective_order=False; output 0-1.
SacreBLEU significance uses the exact same cleaned inputs and BLEU configuration,
with the equivalent 0-100 scale. chrF: six character orders, beta=2, no word n-grams,
0-100. ROUGE is the existing local macro-F1 implementation (ASCII tokens;
rougeLsum aliases rougeL), not a newly substituted library implementation.
CharacTER is the existing `cer` package's word-shift/character-edit measure,
hypothesis-length normalized and capped at 1; empty hypotheses receive 1.
It is not ordinary reference-normalized character error rate.

Every model/source compares Ge-Gs, Ge-Gl and Gs-Gl in each view for BLEU and chrF.
100,000 paired bootstrap resamples, seed 20260917, one Holm family of 120 tests.
For zero observed effects (absolute difference <= 1e-12), use conservative
`p_raw=1` before Holm. SacreBLEU's strict-tail calculation can otherwise give a
tiny p-value for identical systems. Its unmodified p-value is separately retained
as `p_sacrebleu`, and `zero_effect_guard` records the adjustment. This rule is
fixed before the production comparisons and tested on identical predictions.
No ROUGE/CharacTER tests or new grammar-free baseline. Significance concerns
differences among grammar conditions, not whether grammar helps relative to no grammar.
Non-significance does not establish equivalence.

## Artifacts

- `provenance.json`: original file SHA256 hashes, rules, settings, dependency versions.
- `audit.json`: complete/partial/invalid states, counts, prompt checks, retries and limitations.
- `records/`: raw/extracted views and exact spans; no original result is replaced.
- `metrics/`: scores with all records plus historical-denominator and extraction effects separately.
- `analysis/`: 120 test results, p-values, Holm adjustment, error reports and resumable pair checkpoints.
- `condition_matrix.tsv` and `CLOSURE_REPORT.md`: English handover generated from validated artifacts.

An audit can verify saved controls, not unrecorded historical runtime settings or
responses from previous overwritten retry rounds. Four Qwen3.5 rerun records are
explicitly identified. Actual Gl lengths are reported rather than described as
uniformly 100K tokens. The main 351-condition matrix is unchanged.

`*_status.json` files describe the latest stage state. `*_error.json` files,
when present, retain the last diagnostic even if a later rerun succeeds; consult
the status file and validated report, not an old error file alone.
