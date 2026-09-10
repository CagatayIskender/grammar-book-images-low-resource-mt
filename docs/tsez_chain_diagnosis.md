# Tsez Chain Diagnosis (2026-09-10)

## Confirmed Findings

1. The efficient-attention GQA compatibility issue was fixed and GPU-tested.
   Recent Tsez jobs fit one H100 in FP32, without changing the nine page images.
2. Both revised chain preflights still had one truncated output out of three.
   Some format-valid glosses also copied source words or repeated grammatical
   labels. A valid two-line response does not prove a correct gloss/translation.
3. The selected `ddo-train-track1-covered` file hides all glosses. The local
   `ddo-train-track1-uncovered` file has actual training glosses. All 3,558
   source/reference pairs match in order, including the same 21 support examples.
   The earlier diagnosis checked only the covered file and was incomplete.
   Natugu (791) and Lezgi (701) also have aligned uncovered training files;
   no corresponding uncovered Gitksan training file was found locally.
4. Current generation forces `do_sample=False`. Official model cards recommend
   a different VL/non-thinking policy: sampling at temperature 0.7, top_p 0.8,
   top_k 20, repetition_penalty 1.0 and presence_penalty 1.5. This is a supported
   candidate, not proof that greedy decoding caused every failure. The HF
   versions here do not expose presence_penalty in generation configuration;
   the diagnostic applies it through a tested generated-token-only logits
   processor. It is not silently ignored or substituted with repetition_penalty.

Primary sources checked on 2026-09-10:
- https://huggingface.co/Qwen/Qwen3.5-9B (non-thinking general tasks)
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct (VL generation hyperparameters)

## Fixed Diagnostic Design

No pending production/preflight code or settings were changed. Diagnostic code:
`runners/diagnose_tsez_chain.py`; frozen manifest and outputs:
`results/diagnostics/tsez_chain_support_decoding_v1/`.

| Arm | Training support gloss | Decoding |
| --- | --- | --- |
| covered_greedy | absent | existing greedy |
| uncovered_greedy | real training gloss | existing greedy |
| covered_sample | absent | manufacturer VL/non-thinking policy |
| uncovered_sample | real training gloss | manufacturer VL/non-thinking policy |

All arms retain the same revised prompt builder, same 21 training sentences,
same nine unmodified images, one H100, FP32, disabled thinking, 512-token cap.
There are no output retries in this diagnostic, equally across all four arms.
The output limit is deliberately not increased while testing repetition.

Eight development examples are chosen by a fixed rule: three longest eligible
sources and five seeded random eligible sources. Exact training/test-source
overlaps and duplicate development sources are excluded. Selection never reads
reference/gloss content. Selected development indices are
40, 48, 62, 85, 131, 307, 339, 352 (zero-based). Seed: 20260910 plus dev index.
Development gold gloss/reference are used for evaluation, never input prompts.
Real training glosses are not synthetic labels or test-answer leakage.

Each model makes 32 generations with one loaded model. Failed formatting,
truncation, raw outputs and repeated-span warnings are retained. Translation
and gloss chrF are diagnostic only; these eight records are not full-test scores.
A repetition warning is a heuristic and must be reviewed, not treated as proof.
The script never approves production automatically, even if all records parse.

## Jobs and Verification

- 5779107: Qwen3 diagnostic, one H100, one-hour limit.
- 5779108: Qwen3.5 diagnostic, one H100, one-hour limit.
- Four diagnostic tests passed in each Qwen environment, including support
  alignment, development selection, and presence-penalty semantics.
- Eleven existing main-suite tests (including shell syntax) passed.
- Code/input/image hashes are frozen in the manifest; a changed dependency
  aborts the diagnostic rather than mixing settings in resumed results.

## Decision and Thesis Boundary

Do not claim a definite fix before the GPU comparisons finish. Check truncation,
empty outputs, repetition warnings, actual glosses and translation metrics
together. A candidate that merely fills output fields is insufficient.
If no arm gives usable outputs, retain Tsez explicit-chain as an unsuccessful
condition/limitation instead of further tuning against the final test sentences.
If an arm is usable, apply it as a documented new setting and run the full test
once; report all remaining invalid outputs rather than dropping hard examples.
Success on eight development sentences cannot guarantee success on all 445 tests.

Adding training glosses changes the available supervision, not just formatting.
Results using them must not be described as a prompt-only comparison against
covered-support Shot runs. Either label that difference explicitly or run
matched-supervision controls if that particular causal claim is required.
Existing successful results need not be globally rerun to retain their original
experimental meaning. API/MTOB, existing results, and completed XCOMET scores
are unchanged. No GitHub push was performed.
