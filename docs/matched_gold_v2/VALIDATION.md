# Local Validation Before Full Submission

No Slurm smoke/preflight jobs were used. Local checks are not a guarantee that
longer gold-support prompts will fit every GPU case or produce every translation.

- Prepared 351 new configurations, with 21 nonempty aligned gold support glosses
  each (7,371 repeated support slots; four distinct language support sets).
- Generated 48 full job scripts; each condition occurs exactly once in their
  groups. No historical prediction is copied into the corrected output paths.
- Eleven unit/integration tests cover every condition's actual
  constructed user prompt, reject blank/tampered support, check baseline/material
  consistency, and verify that gold test annotations do not enter the prompt.
- One original Lezgi support source also occurs in the test set. It is excluded
  and replaced by the next eligible gold training example in all Lezgi conditions.
  All four corrected support sets are disjoint from their test sources. Source
  positions and replacement indices are recorded in the support manifests.
- An offline fake-backend integration test exercises the real group runner,
  prompt construction, result recording and resume. It sends no API requests,
  loads no model and starts no Slurm job. Resume does not duplicate records.
- Qwen cohorts remain 37/87/99/445; Gemini remains 37/87/99/99 by language.
- ModelGloss predicted test glosses, all grammar material paths, image counts,
  image resolution, output cap, sampling settings and seed policy are unchanged.
- Every local generation job requests one H100 and uses the existing FP32,
  efficient-attention backend with no offload or quantization.
- A source/config/code hash check runs before model loading. Every recorded
  response includes its actual system/user prompt and support hash. Scoring
  checks those fields against a freshly reconstructed gold-support prompt.
- Historical configuration/result/metric hashes are preserved and checked.
- XCOMET is separate from generation. The previous full scorer took 02:31:04;
  the new limit is 06:00:00. Previous native/common99 analyses took 01:38:27 and
  01:31:33; the new combined CPU job requests 05:00:00.

Completion must subsequently be established from `status.json`, Slurm states,
full output counts, correct prompt evidence, XCOMET provenance and complete
native/common99 analyses. Empty model predictions remain in the denominator;
they are not silently replaced, omitted, or fixed by changing the prompt.
