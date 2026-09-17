# Implementation Status

## Completed Locally

- Organized source PDFs, frozen materials, runners, jobs, results and metrics.
- Verified 986 protected material/result/metric files against migration hashes.
- Verified the nine MTOB immutable inputs against its original baseline hashes.
- Generated 210 explicit current conditions: 70 Shot, 70 ModelGloss, 70 ChainGloss.
- Scoped this repair campaign to 70 ChainGloss conditions and five output repairs.
- Prepared and dataset-validated 22 existing Lezgi outputs for XCOMET-only scoring.
- Passed nine main-suite unit/static tests and fourteen existing MTOB tests.
- Confirmed model-specific prompt imports and all legacy image/text input paths.
- Loaded both cached Qwen processors without GPU/model weights and confirmed
  the disabled-thinking chat templates used by the generation entry point.
- Checked every active shell script with bash -n; no multi-GPU request remains.

## Preflight Diagnosis (2026-09-09)

All 15 original preflights (5777740-5777754) failed with zero of three records:
`No available kernel. Aborting execution.` The logs report unequal query/KV
heads (Qwen3: 32/8; Qwen3.5: 16/4). The torch 2.7 efficient SDPA backend cannot
execute the native GQA dispatch chosen by the installed Transformers versions.
This failure is not evidence of a CUDA OOM or a successful memory validation.

The isolated audited runner now uses `efficient_gqa_v1`. It expands K/V with the
installed Transformers `repeat_kv` helper, then delegates masking, causality,
scaling and output layout to the installed SDPA integration. The adapter is
scoped to generation and restores the registry afterward; no environment/library
files were patched. That kernel correction preserved FP32, disabled thinking,
one H100, image bytes/counts and prompts. There is no math/precision/offload
fallback in generation.

Three attention tests passed in each Qwen environment, including 12 numerical
cases against native-GQA math attention, registry restoration and FP32 guards.
The nine existing main-suite tests and two preflight-gate tests also passed.
Each new GPU preflight first checks 12 small FP32 efficient-kernel cases before
loading weights. Jobs 5777849 and 5777850 passed those GPU checks and generated
all three test records without OOM (peak allocated memory about 43.3 and 42.4
GiB). They nevertheless FAILED output validation: Qwen3 had one truncated output;
Qwen3.5 had two truncated outputs and one empty gloss/translation. Runtime was
3:47 and 3:46 respectively. This establishes feasibility for these tested inputs,
not universal memory feasibility or usable chain-gloss quality.

Original `_efficient` preflight artifacts are retained. New artifacts use
`_efficient_gqa_v1`. The production gate checks code/helper, prompt, input and
dataset provenance plus all three nonempty, error-free preflight records.

## Chain Prompt Revision

The 21 support examples all have empty gloss fields in the actual training
files. The old chain prompt printed those blank fields, mixed example/output
translation labels, and did not clearly specify a morpheme-gloss task. Existing
Tsez translation-only outputs also contain repetitions and cut-off predictions
for the same longest sentences; old completion does not establish output quality.

`gloss_prompt_v2` omits missing gloss lines, keeps any supplied nonempty gloss,
uses `FINAL_TRANSLATION:` consistently, and explicitly requests English lexical
meanings plus grammatical labels in source order. Unknown morphemes may use `?`;
no gold annotations, synthetic support glosses or test references were added.
It applies only to the 70 explicit chain-gloss conditions. Shot, ModelGloss,
legacy, API and MTOB prompt builders are untouched.

New chain preflight files end in `_efficient_gqa_v1_gloss_prompt_v2`; previous
attempts remain intact. The same longest three sentences and the same nine Tsez
pages are used for comparison with 5777849/5777850. Attention, FP32, greedy
decoding, 512 output tokens, retries and thinking controls are unchanged. New
provenance records the prompt revision and the gate rejects earlier revisions.
Eleven main-suite tests passed in each Qwen environment, plus three gate tests.
The tests cover all 70 chain configs, no blank gloss examples, consistent labels,
and no test-reference/gloss leakage. Output quality still needs GPU validation.

This correction does not require global reruns of successful Shot/ModelGloss,
API or MTOB experiments, or rescoring unchanged predictions. The planned four
empty Qwen3 outputs and incomplete Tsez Qwen3.5 ModelGloss output remain separate
repairs. The 22 Lezgi XCOMET scores have now been completed and verified.
Historical duplicate Qwen3.5
shot/chain predictions must not be reported as distinct explicit-chain evidence.
Known repetitive/truncated historical outputs require quality reporting or a
separately documented targeted rerun, not silent cleanup of published scores.

## Verified Outcomes (2026-09-10)

- 5777755 COMPLETED in 13:35. All 22 Lezgi files were scored, rejected=0.
  A fresh score_verified.py --dry-run verified current prediction hashes and
  complete score provenance for every target: pending=0, rejected=0.
- 5778169 (Qwen3 Tsez) FAILED in 4:45; three records, one truncated output.
- 5778170 (Qwen3.5 Tsez) FAILED in 3:28; three records, one truncated output.
  Qwen3.5's format-error count fell from three to one, but valid formatting is
  not a correctness measure. Repetitions and questionable glosses remain in
  some format-valid records. Both jobs fit one H100 without OOM.

Tsez chain full generation remains blocked. Do not relax the output gate or
treat this three-sentence comparison as translation-quality evidence for a
whole language. No further prompt/decoding change was made in this continuation.
Other languages and the separate Shot/ModelGloss repairs have independent
preflights; their validation can proceed without assuming Tsez chain succeeded.

## Historical Submission Snapshot

| Jobs | Purpose | GPUs per job | Requested time |
| --- | --- | --- | --- |
| 5779082-5779083 | Gitksan PDF1 chain, Qwen3 / Qwen3.5 | 1 H100 | 1 hour |
| 5779084-5779085 | Gitksan PDF2 chain, Qwen3 / Qwen3.5 | 1 H100 | 1 hour |
| 5779086-5779087 | Lezgi chain, Qwen3 / Qwen3.5 | 1 H100 | 1 hour |
| 5779088-5779089 | Natugu chain, Qwen3 / Qwen3.5 | 1 H100 | 1 hour |
| 5779090-5779091 | Natugu Qwen3 Shot / ModelGloss repairs | 1 H100 | 1 hour |
| 5779092-5779093 | Tsez Qwen3 Shot / ModelGloss repairs | 1 H100 | 1 hour |
| 5779094 | Tsez Qwen3.5 ModelGloss repair | 1 H100 | 1 hour |

At the latest submission verification these 13 jobs were PENDING.
This is a historical snapshot, not a live queue view. Submission records are in
docs/submissions. All are three-sentence preflights with unchanged image sets;
none is full generation. Each full condition still requires its own applicable
current successful preflight. The two failed Tsez chain preflights were not
blindly resubmitted.

## Remaining After Preflight

Tsez-specific diagnosis is documented in `docs/tsez_chain_diagnosis.md`.
Jobs 5779107/5779108 independently compare covered/uncovered training support
and greedy/manufacturer sampling on eight disjoint development examples.
They are one-H100 FP32 diagnostics, not full experiments or automatic production
approval. Pending jobs and existing prompt/decoding defaults were not changed.

At that snapshot full generation had NOT been submitted, and the five missing
outputs were not complete. The submit tool blocks production where the corresponding
current FP32 preflight has not passed. Do not infer full-test results from the
three-sentence preflight files.

Historical greedy-runner commands (not the new sampled candidate):

```bash
python3 scripts/submit_jobs.py --family chain_gloss_v2 --model qwen3 --group natugu --submit
python3 scripts/submit_jobs.py --family chain_gloss_v2 --model qwen35 --group natugu --submit
python3 scripts/submit_jobs.py --repairs --submit
```

Use the matching language/PDF group wrappers for the other chain conditions to
stay within the Slurm submission limit. After all required outputs are complete,
submit scripts/scoring/run_repair_xcomet.sh. Failed one-GPU FP32 preflights must
be investigated; no automatic GPU-count, image-resolution or precision fallback
is permitted. No API generation or GitHub push was performed.

## Continuation After Diagnostic Completion

See `docs/sampled_decoding_validation.md` for the measured four-arm comparison,
the separate `sampled_v1` runner and the current submission list. The frozen
greedy runner and diagnostic files were not edited.

Four independently verified full repairs were submitted as 5780306-5780309.
The failed Tsez Qwen3 Shot repair was correctly blocked by its original gate.
Thirteen sampled preflights were submitted as 5780321-5780333, each requesting
one H100 for one hour. Full sampled Chain generation remains blocked pending
successful exact-condition validation. These are submission records, not claims
that full outputs or final metrics have completed.

## Continuation (2026-09-11)

- Sampled preflights: 12/13 passed with current provenance and result hashes.
  Qwen3.5 Tsez selected-page Chain failed: the longest source repeats gloss
  fragments until 512 tokens in both attempts. This is not CUDA OOM.
- Full ModelGloss repairs completed with matched dataset order and no recorded
  output errors: Natugu Qwen3 99/99, Tsez Qwen3 445/445, Tsez Qwen3.5 445/445.
- Natugu Qwen3 Shot wrote 99 records but two have truncated, repeated translations.
  It is not an error-free result. Its failure cancelled dependent scorer 5780342.
- Scoring was separated: 5784430 now scores only the three complete ModelGloss
  outputs, requesting one H100 for five hours. No generation rerun for these.
- Submitted 5784390-5784429: 39 Chain conditions plus the Tsez Qwen3 Shot repair.
  Twelve already passed sampled validation; 28 must first pass their exact
  condition's preflight in the job, then the runner independently checks that
  gate before full generation. `set -e` stops failed validations.
- Thirty additional Chain conditions remain unsubmitted for the next bounded
  batch. The Qwen3.5 Tsez selected-page Chain is separately blocked and is not
  silently retried. Natugu Shot's two failed greedy outputs are preserved;
  no mixed-decoding row replacement was performed.

Use `python3 scripts/submit_sampled_campaign.py --validate-first --max-jobs 20`
to list the next batch. Add `--submit` only when queue capacity is available.
The tool skips queued/completed conditions and blocks failed preflights or
full outputs with errors. All jobs remain one H100, FP32, unchanged images;
Tsez requests ten hours, other generation jobs four hours. New results use
`_sampled_v1`. Generated shell scripts passed `bash -n`; sampled runner tests
and campaign gate checks passed. No model runner was changed in this continuation.

Successful operational checks do not establish gloss correctness. Keep the
documented decoding confound when comparing sampled Chain to old greedy Shot.

## September 13 Continuation and Baselines

The 40-job batch produced 24 error-free full outputs and five full outputs with
recorded generation failures. Eleven jobs stopped with zero records at model
load on `lrz-hgx-h100-026`. Logs identify other processes 866820/866821 occupying
approximately 76 GiB. This is distinct from an intrinsic context attention OOM;
the process owner/cause has not been established. Scorer 5784430 completed in
3:37 and filled the three previous ModelGloss repairs.

The 11 zero-record load failures were archived with a hash manifest under
`archive/model_load_oom_1789336055790076124/`. Their retry excludes that node,
without changing FP32, image content or decoding. Jobs 5788166-5788206 comprise
these 11 retries and the 30 previously unsubmitted conditions. The five recorded
full-generation failures and the failed Tsez Qwen3.5 selected-page preflight
remain blocked, not silently regenerated or replaced.

At the user's additional request, four Qwen3.5 baseline suites were submitted:
5788224 (Gitksan), 5788225 (Lezgi), 5788226 (Natugu), 5788227 (Tsez). Each suite
contains Shot/ModelGloss/explicit Chain under greedy and sampled decoding, with
no grammar context. Exactly one H100 per suite, sequential model reuse, FP32.
Every condition needs its own initial preflight. See
`docs/qwen35_baseline_significance.md` and the submission JSON for provenance.

After those suites terminate, CPU job 5788229 applies paired sacreBLEU bootstrap
tests to available same-model/same-policy Qwen3.5 pairs (and the Qwen3 snapshot),
and one-H100 job 5788230 scores complete error-free baseline/sampled outputs with
XCOMET. Partial baselines and failed outputs remain explicit exclusions. The
analysis reports baseline coverage, not just successful tests. Latest queue
check: all 47 submitted/dependent jobs were pending; no new baseline outputs
existed yet. These are not claims of finished baseline metrics.

The initial Qwen3 analysis completed: 326 metric comparisons, 10,000 paired
bootstrap resamples, BLEU and chrF++, raw and Holm-adjusted p-values. Twenty-one
positive differences have unadjusted p < .05, but zero positive differences
survive the declared broad Holm correction. One negative difference survives.
Do not equate a non-significant test with equivalence or proof of no effect.
The report, raw tables and input hashes are under
`docs/statistical_significance/2026-09-13/`.
