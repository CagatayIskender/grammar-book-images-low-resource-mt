# Matched Grammar-Context Ablations

## Final Handover Status

As verified on 2026-09-17, all 351 primary conditions (Qwen3, Qwen3.5 and Gemini)
are complete, with current BLEU/chrF++, XCOMET-XL and paired statistical results.
Luna remains in historical machine-readable catalogs but is outside primary thesis
scope. Start with the [final thesis guide](../../THESIS_GUIDE.md) and
[completion/comparability audit](completion_audit_2026-09-17.md).
Campaign preparation and budget discussions below are protocol history, not a
statement that further generation is required. English matrices now display the
three primary models; saved statistical correction families retain the original
four-model plan. See the guide before filtering results.

## Protocol

This suite measures grammar-context changes against a matched, grammar-free
control for each model and prompting method. It preserves historical outputs in
their existing families. Active configurations are frozen in `configs/matched_v1`.
These are adapted GRAMMAMT strategies, not a verbatim paper replication.

## Cohorts

| Model | Gitksan | Lezgi | Natugu | Tsez |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 | 37 | 87 | 99 | 445 |
| Qwen3.5 | 37 | 87 | 99 | 445 |
| Gemini 2.5 Flash Lite | 37 | 87 | 99 | 99 |
| GPT-5.6 Luna | 37 | 87 | 99 | 99 |

API Tsez uses the first 99 test rows, not a random or response-selected sample.
The two Qwen models retain all 445 rows. Cross-model Tsez comparisons use the
same first 99 rows from all models, without new Qwen generation. Native Qwen
445-row scores must not be directly compared with API 99-row scores.
All four languages retain the same first 21 support examples per language.
Gitksan's two PDFs share one language/method baseline; they are not independent
datasets or duplicated baseline runs.

Source-count audit: Lezgi's 87 rows contain 85 unique source strings and 86
unique source/reference pairs. Gitksan, Natugu and both Tsez cohort sizes have
unique source strings in this extraction. Lezgi rows are kept in the canonical
test set; sentence-bootstrap significance assumes row-level exchangeability,
so interpret its duplicated-source observations with caution.

## Matrix and Controls

There are 117 planned conditions per model, 468 total:

- Three methods: Gloss-shot (`shot`), explicit Chain-gloss (`chain_gloss`),
  and predicted-gloss input (`modelgloss`).
- Five sources: Gitksan Brown, Gitksan Rigsby, Lezgi, Natugu and Tsez.
- Six materials per source: cheat sheet TXT/JPG, summary tables TXT/JPG,
  summary text TXT, selected PDF-page JPGs.
- Five additional Lezgi Cyrillic materials per method (no separate PDF pages).
- One grammar-free baseline per language and method: 12 per model.

Within a model/method, source/reference rows, support examples, predicted glosses,
system instructions, decoding settings and output parsing stay fixed. Grammar
text or image context and its accompanying context instruction are the ablation.
Gloss-shot controls still contain glossed support examples; they are not
gloss-free few-shot controls. ModelGloss keeps the same cached predicted glosses
with and without grammar. Their quality is not independently re-estimated here.

The original paper's prompts are adapted: for example, local/API ModelGloss does
not include the paper's predicted-gloss error warning. Do not call historical
baseline scores matched to this new family without a separate provenance check.
The API non-chain template is reused unchanged; explicit Chain-gloss has its
own fixed prompt, including a gloss followed by the final translation.

## Generation Policy

Local Qwen generation requests exactly one H100, FP32, no CPU offload,
quantization or image reduction. Both Qwen models use the existing
development-validated sampled configuration: temperature 0.7, top-p 0.8, top-k 20,
min-p 0, presence penalty 1.5, repetition penalty 1, 512 new tokens.
The seed is `20260910 + 2 * original_test_index`. Thinking is disabled.
These are fixed experimental settings, not claimed to be every model's defaults.

Gemini uses temperature 0, seed 42, 512 new tokens, reasoning effort `none`,
and the Google provider. Luna uses seed 42, 512 new tokens, effort `none`;
temperature is omitted because the endpoint does not support it. Mandatory
reasoning or returned reasoning fields/tokens stop the API backend. Models are
not silently substituted. Cross-model differences therefore describe the
configured systems, not architecture alone.

One semantic response is retained per sentence. There are no format-repair or
best-guess reprompts, no response-dependent decoding changes, and no retries
that select the best translation. Identical transport requests may be retried.
Raw output and flags are retained. A produced final translation can be extracted
despite malformed/missing gloss formatting. Empty, truncated, refused or
unextractable outputs remain errors in the full denominator. Reasoning output
is not treated as a translation. A GPU-memory failure marks a condition
unavailable instead of silently changing precision, resolution or sample count.

Matched imports are copies, not changes to historical files. Audited Gemini
Shot/ModelGloss results are reused where settings, data and material hashes
match. Some historical API records lack per-request prompt hashes and finish
reasons; this is a provenance limitation. Qwen Chain imports retain only verified
first attempts with matching decoding, source, support, material, seed and code
provenance. Responses from earlier prompt-repair attempts are not imported.

## Budgets, Jobs and Resume

Luna has a user-approved **4 USD total budget** for this campaign, not per job.
Its single worker runs controls before material conditions. Unaffordable
conditions remain incomplete, not zero-scoring completed experiments. The local
reservation ledger is conservative, reconciles returned usage costs, and
stops before a request would exceed its available budget. It is not an
OpenRouter account-wide billing limit. Gemini's local campaign guard is 10 USD.

API workers are serialized per model through Slurm dependencies. Initial
concurrent Gemini jobs collided while writing shared metadata/budget files;
the replacement jobs resume existing rows and do not change prompts. Cross-node
file locks alone were insufficient to prevent those observed collisions.
Use the submit helper, not simultaneous manual API job submissions.

Generation limits are workload-specific (1, 3, 4, 6 or 10 hours); see
`runtime_adjustments.md` for historical evidence and safety margins. Full Tsez
Qwen conditions are grouped in batches of at most three. A worker checkpoints
10 minutes before a one-hour limit, otherwise 30 minutes before the limit, with exit code 75;
rerun the same submit helper to resume missing rows. A zero Slurm exit status
alone does not certify a complete matrix: inspect the condition catalog,
resource-unavailable markers and expected record counts.

Commands from the GRAMMAMT root:

```bash
venv/bin/python scripts/status_matched.py --validate
venv/bin/python scripts/submit_matched_jobs.py --models qwen3 qwen35
venv/bin/python scripts/submit_matched_jobs.py --models qwen3 qwen35 --submit
venv/bin/python scripts/submit_matched_jobs.py --models gemini25flashlite --submit
MATCHED_API_BUDGET_USD=4 venv/bin/python scripts/submit_matched_jobs.py --models gpt56luna --allow-luna --submit
venv/bin/python scripts/submit_matched_scoring.py --submit
```

The generation submitter skips queued names and complete verified outputs. Do
not restart a running group by calling its Python runner directly. API secrets
come from the existing protected environment loader; no key belongs in configs.

## Scoring and Statistical Tests

XCOMET-XL runs separately on one H100, with a 14-hour request. It requires full
configured cohorts and source matches, and skips a score only when its result
hash, model and record count match. It is not the paper's XCOMET-XXL.

SacreBLEU paired bootstrap tests use BLEU (13a) and chrF++ with 100,000 resamples,
fixed seed 20260915. Material-versus-matched-baseline comparisons use a
predefined global Holm family; unavailable planned tests count conservatively
as p=1. TXT/JPG paired comparisons form a separate Holm family. The higher
bootstrap count supplies p-value resolution for the large planned family;
the earlier 100-resample local execution is a software smoke test only.
Generation settings were fixed before this significance analysis, but earlier
development used the same datasets: this is not an untouched confirmatory test.

Native and common99 analyses have separate CPU jobs, each requesting 10 hours:

- `docs/matched_v1/analysis_native`: within-model native-cohort results.
- `docs/matched_v1/analysis_common99`: matched-size cross-model view.

These views overlap; do not treat them as independent replications. Use
common99 as the primary cross-model view and native results as within-model
supplementary analysis. Scores are based on all configured rows, including
empty translations. Incomplete conditions are explicitly excluded, not scored
on a convenient intersection. Confidence intervals are per-system, not
paired-delta intervals. Non-significance is not proof of equivalence.

TXT/JPG results alone do not isolate vision ability unless textual content is
also equivalent. Some source pairs are not certified identical. No human
adequacy review, causal model-interaction test, minimum practically important
difference or independent sentence/document sampling validation is supplied.
Grammar-book/test overlap and duplicated test sentences remain dataset-level
limitations; sample size here counts test rows, not necessarily independent
linguistic examples. Report these limits alongside thesis claims.

Evaluation jobs wait for currently queued generation jobs with `afterany` so
failures can be reported. They do not magically complete budget-stopped,
resource-unavailable or checkpointed conditions. After later continuations,
resubmit evaluation to refresh the report; result hashes guard stale scores.

## Files and Reproduction

- `configs/matched_v1/catalog.json`: planned conditions and frozen configs.
- `docs/matched_v1/experiment_catalog.tsv`: current counts, states and paths.
- `docs/matched_v1/status.json`: timestamped validation snapshot.
- `docs/matched_v1/submissions.json`: generation receipts, including historical
  cancelled/replaced IDs; consult Slurm for current state.
- `docs/matched_v1/scoring_submissions.json`: evaluation dependencies and IDs.
- `results/matched_v1` and `metrics/matched_v1`: only this protocol's artifacts.
- `docs/matched_v1/*_budget.json`: cumulative API accounting, no credentials.
- `docs/matched_v1/tsez_scope_correction.json`: records the user's clarification
  restoring the 42 Qwen Tsez conditions to 445 while API conditions stay at 99.

Use the frozen configurations for an existing run. A fresh, isolated recreation
uses `runners/matched/prepare.py`, followed by
`scripts/restore_matched_qwen_tsez_full.py` and
`scripts/prepare_matched_jobs.py`. The original preparation code was frozen
before the cohort clarification; the correction step is required. Do not
regenerate active configs or edit fingerprinted code during queued/running jobs.
The correction utility rejects replacing newly generated Tsez records.

Validation:

```bash
venv/bin/python -m unittest discover -s tests -p test_matched_protocol.py -v
venv/bin/python scripts/status_matched.py --validate
venv/bin/python runners/matched/score.py
```

The last command computes CPU metrics and lists incomplete conditions. Neither
it nor the tests submit jobs or make API translation requests.
