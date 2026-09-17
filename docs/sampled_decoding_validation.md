# Sampled Decoding Validation (2026-09-10)

## Completed Diagnostic

The frozen 2x2 diagnostic used the same eight development sentences, nine
unaltered Tsez JPGs, FP32, one H100, disabled thinking, and a 512-token cap.
The diagnostic manifest and original runner files remain unchanged.

| Model | Support / decoding | Invalid / 8 | Truncated / 8 | Repetition flags | Translation chrF++ |
| --- | --- | --- | --- | --- | --- |
| Qwen3 | Covered / greedy | 3 | 3 | 2 | 13.324 |
| Qwen3 | Uncovered / greedy | 5 | 5 | 1 | 9.288 |
| Qwen3 | Covered / sample | 0 | 0 | 0 | 18.589 |
| Qwen3 | Uncovered / sample | 2 | 2 | 0 | 15.762 |
| Qwen3.5 | Covered / greedy | 3 | 3 | 3 | 8.939 |
| Qwen3.5 | Uncovered / greedy | 4 | 4 | 3 | 7.119 |
| Qwen3.5 | Covered / sample | 0 | 0 | 0 | 13.398 |
| Qwen3.5 | Uncovered / sample | 1 | 1 | 0 | 14.865 |

Jobs 5779107 and 5779108 completed in 21:31 and 12:41. Peak allocated GPU
memory was approximately 43.78 and 42.77 GiB. These are not full-test scores.
The experiment varied the entire decoding policy, so it does not isolate the
effect of sampling from the presence penalty.

Manual inspection still finds source copying in Qwen3 glosses and incorrect
translations in both models. Successful parsing is not linguistic validation.
Do not describe these findings as proof that the translation problem is solved.

## Candidate Policy

`runners/run_sampled_context.py` leaves the frozen original runner untouched.
It uses the original first 21 covered training examples, not newly supplied
gold glosses. No source image, prompt content, precision, or attention backend
changes. The policy is explicitly versioned `sampled_v1`:

- Sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0.
- Repetition penalty 1.0; generated-token-only presence penalty 1.5.
- Maximum 512 generated tokens; up to two format attempts, as in the audited runner.
- Seed `20260910 + 2 * original_test_index + attempt` makes resume order independent.
- Repeated spans, truncation, malformed output and thinking tags cannot pass validation.
- Raw outputs, seeds, attempts and prompt hashes are retained; no gold gloss substitution.
- Full output and metric filenames have `_sampled_v1`; greedy files are preserved.
- Each full sampled condition requires its own current three-sentence preflight,
  verified against source, configuration, code, policy and result hashes.

This policy is the tested image policy, deliberately also being checked on two
Tsez text conditions. It is not a claim that the image recommendation is optimal
for text-only tasks. Seeded GPU sampling is not guaranteed bitwise reproducible
across hardware or library versions.

The old `scripts/submit_jobs.py --family chain_gloss_v2` still submits the frozen
greedy runner. Do NOT use that command to launch the sampled campaign.

## Submitted Work

Four full repairs passed their original greedy preflights and retain those settings:

| Job | Condition | Requested time |
| --- | --- | --- |
| 5780306 | Natugu Qwen3 Shot selected pages | 4 hours |
| 5780307 | Natugu Qwen3 ModelGloss selected pages | 4 hours |
| 5780308 | Tsez Qwen3 ModelGloss selected pages | 10 hours |
| 5780309 | Tsez Qwen3.5 ModelGloss selected pages | 10 hours |

XCOMET job **5780342** requests one H100 for five hours and depends on
`afterok:5780306:5780307:5780308:5780309`, with invalid-dependency cancellation.
It uses `configs/verified_repair_scoring_targets.json` and scores only these four
full outputs. Generation and scoring therefore cannot share a GPU allocation.

Thirteen sampled preflights (5780321-5780333) request one hour each:
ten model/source selected-page Chain combinations, two Tsez summary-text Chain
combinations, and the failed Tsez Qwen3 Shot selected-page repair. Each uses the
three longest test sources for operational stress testing, not to choose settings
by test translation scores. No full sampled Chain run was submitted.

```bash
# Prepare/list validation scripts without submitting:
python3 scripts/submit_sampled_preflights.py
# Submit only checks not already queued or verified:
python3 scripts/submit_sampled_preflights.py --submit
```

All GPU jobs request exactly one H100. No offload, reduced precision, image
removal, or downscaling was introduced. Historical Shot/ModelGloss results do
not need blanket regeneration. A future sampled Chain versus greedy Shot
comparison has a decoding confound and must be reported as such, or supplied
with a separately authorized matched-decoding control. Do not silently call it
a prompt-only causal comparison.
