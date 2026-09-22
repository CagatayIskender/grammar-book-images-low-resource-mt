# Gold Support Correction: matched_gold_v2

Status on 22 September 2026: all 351 conditions and 40,731 records are complete.
XL/XXL scoring and all planned statistics completed. Use the [final handover](../../reports/matched_gold_v2/README.md)
and [completion/comparability audit](../../reports/matched_gold_v2/COMPLETION_AUDIT.md), not old pending-job snapshots.
Share with the documented caveats: Lezgi has invalid references and descriptive
84/83-row sensitivities; complete records do not imply successful translations.

## Why This Revision Exists

The historical matched_v1 configurations contain 21 empty support gloss fields
in every primary condition: 351 conditions and 7,371 repeated support slots.
The earlier documentation claiming gold/glossed support examples was incorrect.
The covered training input files deliberately suppress glosses. A template label
and a complete result count do not prove that the intended information was sent.

Historical configurations, results and metrics remain unchanged. Their support
sets are translation examples without gloss demonstrations. Shot cannot be
reported as original GRAMMAMT Gloss-shot. Historical Chain-gloss still requests a
test gloss, and ModelGloss still supplies a predicted test gloss, but neither
has gold-gloss support demonstrations.

## Corrected Scope

- 351 conditions: Qwen3, Qwen3.5 and Gemini 2.5 Flash Lite; 117 each.
- All 36 baselines and 315 material conditions are regenerated; no old predictions
  are imported. Gold support changes the prompt for every method.
- The first 21 training examples receive gold annotations, except that any
  source also present in the test set is excluded and replaced by the next
  eligible training example. One original Lezgi support/test overlap was found.
  All Lezgi baseline/material/model conditions receive the same replacement.
  The other languages retain their original 21 sources/translations/order.
- Natugu/Lezgi/Tsez: local train-track1-uncovered files, exact positional source
  matching before the documented Lezgi exclusion/replacement.
- Gitksan: official SIGMORPHON data_v1 training file at pinned commit
  190689ac81935359c69a46463c48e25e63e601f7. License stored alongside it.
- No test gold gloss is supplied. ModelGloss keeps the same cached predicted
  test glosses; Chain-gloss generates its own test gloss.
- Same grammar files/images, image counts/resolutions, prompt templates,
  generation settings, seeds, output cap and failure policy as matched_v1.
- Qwen uses 445 Tsez rows; Gemini uses the same first 99. Other cohort sizes:
  Gitksan 37, Lezgi 87, Natugu 99. Luna remains excluded.
- One H100 per Qwen or XCOMET job; Qwen FP32, no offload or quantization.
  Independent jobs may run concurrently. API jobs need no GPU.
- No smoke or small preflight jobs, as requested. Local tests precede submission.
- Empty/filtered/truncated output remains a recorded failure, not a reason to
  modify the prompt or select another semantic attempt.

## Evidence and Checks

`support/` records complete aligned gold examples, source paths/URLs and hashes.
`prompt_examples/` records a fully instantiated prompt per condition. Generation
also saves the actual system/user prompt and its hash in each output record.
Validation reconstructs the expected prompt and checks that evidence on resume
and scoring. Empty, altered or wrong-language gold support fails before model
loading. Source files, materials and implementation are hash-bound.

`historical_hashes.json` protects previous configuration/result/metric files.
The gold support and predicted test gloss are identical within each corrected
baseline/material pair. The statistical family is defined for three models:
630 planned context-versus-baseline metric tests and 216 TXT/JPG metric tests
per cohort, with separate Holm correction families and 100,000 resamples.
These families are distinct from the historical analysis, which included Luna.

## Commands

From the project root:

```bash
venv/bin/python scripts/finalize_gold_v2.py
venv/bin/python -m unittest discover -s tests -p test_final_gold_v2.py -v
```

The following are historical reproduction commands, not outstanding work.
Do not regenerate frozen configurations or submit the completed campaign merely
to refresh the handover:

```bash
venv/bin/python runners/matched_gold_v2/prepare.py --download-gitksan
venv/bin/python -m unittest discover -s tests -p test_gold_support_v2.py -v
venv/bin/python scripts/submit/matched_gold_v2/submit_all.py
venv/bin/python scripts/submit/matched_gold_v2/submit_all.py --submit
venv/bin/python runners/matched_gold_v2/status.py
```

The submitter sends 48 full generation jobs, then one separate single-H100
XCOMET-XL job after generation terminates, then one CPU job for native/common99
significance after successful scoring. A submission receipt prevents duplicate
submissions and supports resuming after scheduler quota rejection. It does not
automatically retry terminal failed jobs; inspect their state before resubmission.
Gemini jobs are chained serially with afterany dependencies because their shared
budget/model-metadata writer is unsafe under concurrent cluster writes. Qwen jobs
remain independent. `--submit --retry-api` replaces only terminal API submissions,
retains submission history and saved outputs, and updates scoring dependencies.
Gemini shares the existing locked USD 10 campaign spending ledger/cap. Credentials
and spending ledgers remain outside version control. No Luna calls are made.

Results: `results/matched_gold_v2/`; metrics: `metrics/matched_gold_v2/`.
Do not change a frozen configuration or gold source after submission. Do not use
the old thesis completion statement as evidence of completion of this revision.
MTOB outputs are unaffected; the old MTOB/GRAMMAMT comparison is explicitly against
matched_v1, not these corrected baselines. Recompute that supplementary comparison
separately if it is to refer to corrected baselines.

## Initial API Infrastructure Incident

The first parallel Gemini submissions failed on shared budget/model JSON temporary
file renames. Some responses were already saved; these remain immutable and are
skipped on resume, including genuinely empty/truncated model outputs. A response
that failed during budget settlement before result persistence cannot be recovered;
the same unchanged request may be sent again for that missing record. This is an
infrastructure recovery, not response-quality selection. Earlier unpersisted
answers are unavailable, so do not claim complete API response history for these
affected initial jobs. Submission receipts retain the original and replacement
job IDs. Generation code, prompts, seeds and fingerprints were not altered by the
serial scheduling correction.
