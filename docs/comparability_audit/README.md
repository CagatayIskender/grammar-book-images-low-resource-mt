# Within-model comparability audit

**Historical audit, not the final thesis verdict.** This 2026-09-15 snapshot
motivated the matched corrections. Its missing-result and missing-scoring
statements describe the earlier experiment families, not the completed
`matched_v1` matrix. Keep it as methodological history; do not substitute it
for the [final completion and comparability audit](../matched_v1/completion_audit_2026-09-17.md).
Start with the [final thesis guide](../../THESIS_GUIDE.md) for the primary scope,
statistical results and reporting limitations. The original audit evidence below
is retained unchanged.

Snapshot: 2026-09-15. Scope: Qwen3, Qwen3.5, Gemini baseline, legacy,
curated_v1 and chain_gloss_v2 files. Excludes other models, smoke, MTOB and
preflight results. Existing results, metrics and job scripts were not changed.

Run: `venv/bin/python scripts/audit_comparability.py`.
Machine-readable per-file evidence: `audit.json`.

## Completed checks

- Inspected 389 JSONL files, including deprecated Markdown counterparts.
- Validated full expected count, index, source, reference and ordering.
  API Tsez uses the first 99 records; Qwen uses 445. Partial files are rejected.
- Recomputed BLEU (13a) and chrF++ (word_order=2), using sacrebleu 2.6.0,
  for 487 prediction blocks in complete aligned files. No score discrepancies.
- Input hashes were stable during each file's audit. This is a snapshot, not
  a guarantee that running jobs cannot subsequently add output files.
- All 68 Gemini files match reconstructed fingerprints with a 512-token cap,
  temperature 0, seed 42, support_n=21, reasoning effort none, correct model,
  selected context paths and baseline flag. Metadata identifies Google provider.
- All 60 Gemini context files have a complete corresponding baseline. Predicted
  ModelGloss values match sentence by sentence. No empty/error prediction flags.
- Current Gemini prompt builders were executed with 21 real support examples:
  text/image x ordinary/ModelGloss all match the baseline exactly after removal
  of the grammar block and its instruction (4 checks).
- The 72 distinct Gemini material paths resolve through migration_manifest.tsv;
  current contents match migration-time SHA256 values. This is not a per-request
  content hash recorded by the API runner.
- Current Qwen35 Chain prompt builders match the no-grammar baseline after
  context-only removal for text/image x four languages (8 checks).
- 29 complete Qwen35 sampled Chain outputs match complete sampled baselines:
  test/support data, common code hashes, FP32, attention, GPU count, thinking,
  seeds, attempt limit, sampling parameters and 512-token cap. Recorded context
  hashes match current files; recorded code hashes have not changed.

## Verdicts

Gemini: 60 matched context comparisons available against 8 controls. Ordinary
means gloss-shot, not the paper's gloss-free few-shot. Prompt validation is for
current code; historical per-request prompt hashes and token-level backend
execution are not recorded. This limits exact historical reconstruction.

Qwen35 sampled Chain: 26 matched context files have no recorded errors. Three
additional matched files each contain one failed prediction: Gitksan PDF2 pages,
Natugu pages, Lezgi Cyrillic summary text. Keep failures in the denominator for
an end-to-end pipeline evaluation; do not present them as error-free translation
quality. Tsez summary text lacks a complete sampled baseline in this snapshot.
Other planned Tsez conditions may still be missing and are not certified.

Qwen3 sampled Chain: historical baselines are greedy and use a different prompt.
The contrast does not isolate the contribution of grammar context. Do not use
significance testing to disguise that methodological mismatch.

Historical Qwen3/Qwen35: complete predictions and matching recomputed metrics
do not establish equal generation settings or correct historical language labels.
Run-time provenance is absent in many files. Current scripts cannot prove what
an older process actually ran. Keep these comparisons explicitly exploratory.

## Scoring and remaining limitations

- Eight Gemini baselines have no XCOMET yet. Their basic metrics can be compared;
  XCOMET comparisons require separate scoring. Existing Gemini context XCOMET
  scores lack a prediction-content hash in their metric metadata.
- Thirteen completed Qwen35 baseline files have no XCOMET in this snapshot;
  five have a verified XCOMET-XL prediction hash. Do not imply scoring is complete.
- XCOMET was not rerun in this audit. BLEU/chrF++ agreement does not validate it.
- Text/JPG semantic equivalence, PDF/test overlap, hidden API model revisions,
  and human translation accuracy are not certified by this audit. They require
  separate evidence. A first-99 subset is not a random sample of Tsez.
- Failed/empty predictions must not be selectively removed. Incomplete files
  are not full-test results. Old duplicate Shot/Chain blocks are not independent
  Chain conditions. Material-only claims require equal decoding/prompt conditions.
- No new significance test was run here. Statistical and practical significance
  must be assessed separately; absence of significance does not prove equivalence.

## Original GRAMMAMT

Reference: https://aclanthology.org/2025.acl-long.1447.pdf (Appendix L, Figures 8-9).
Historical local gloss-shot follows the paper's system and delimiter structure.
Local ModelGloss omits the paper's warning about possible predicted-gloss errors.
Modern API/Qwen35 prompts change the system message and output marker; label
them as adaptations. The original few-shot baseline does not include glosses.
For this thesis the new controls measure the incremental effect of grammar-book
context over gloss-shot or ModelGloss, not the effect of all grammatical input.
The paper uses XCOMET-XXL, whereas verified local scoring uses XCOMET-XL.
