# GRAMMAMT Experiments

**Final thesis starting point:** [Thesis guide, decisions and limitations](THESIS_GUIDE.md).
The guide links the visual experiment matrix and current queue/condition status.

As verified on 2026-09-17, the primary `matched_v1` matrix is complete:
Qwen3, Qwen3.5 and Gemini each have 117/117 conditions, with validated basic
metrics, XCOMET-XL and paired significance analyses. Luna is excluded from the
primary thesis comparison; its historical artifacts are retained, not deleted.
See the [final completion audit](docs/matched_v1/completion_audit_2026-09-17.md).

For the current thesis comparisons, start with the
[matched-ablation protocol](docs/matched_v1/README.md) and its
[live condition catalog](docs/matched_v1/experiment_catalog.tsv).
Tsez uses 445 examples for Qwen3/Qwen3.5 and the same first 99 for Gemini/Luna;
cross-model analysis uses the common 99. Luna's campaign budget is 4 USD.
The matched family fixes prompts and decoding across materials within each
model/method, and never repairs a prompt based on a failed response.

The earlier [experiment catalog](docs/experiment_catalog.tsv) lists exact
materials, model, language, condition, dataset count, job, result and metric paths.
Machine-readable configurations live in `configs/experiments/`.

## Directory Guide

| Directory | Purpose |
| --- | --- |
| `inputs/grammar_pdfs/<language>/<source>` | Original, unchanged grammar books |
| `materials/legacy` | Earlier screenshots and summaries |
| `materials/curated_v1` | Frozen new-materials collection, including Cyrillic |
| `runners/{baseline,qwen3,qwen35,openrouter}` | Model-specific legacy code and API code |
| `runners/run_audited_context.py` | Earlier repair runner and shared backend |
| `runners/matched` | Frozen matched baseline/material generation and analysis |
| `configs/matched_v1` | Matched cohorts, policies, hashes and job groups |
| `docs/matched_v1` | Matched protocol, status, submissions and significance results |
| `scripts/jobs/<model>` | Individual Slurm generation and preflight jobs |
| `scripts/submit/{curated_v1,chain_gloss_v2,legacy}` | Separate experiment-family submissions |
| `scripts/scoring` | GPU scoring, separate from generation |
| `scripts/generators` | Material generators and catalog/job generator |
| `scripts/local` | Local utilities and PowerShell launchers |
| `results` and `metrics` | Matching family/model/language/source/variant trees |
| `experiments/mtob` | Isolated Ge/Gs/Gl suite, not part of current reruns |
| `slurm_outputs` | Flat Slurm log directory for the main suite |
| `archive` | Superseded launchers, local snapshots and unclassified artifacts |
| `docs/migration_manifest.tsv` | Original-to-moved paths and before/after hashes |
| `docs/migration_followup.tsv` | Additional archive/source moves |

`curated_v1` replaces the ambiguous historical `newmaterials` label. Existing
combined Shot/Chain predictions retain their original record contents and metric
keys. New runs use explicit `shot`, `modelgloss`, and `chain_gloss` filenames.
`chain_gloss_v2` is the corrected explicit-gloss experiment, not a relabeling of
the old chain scores. TXT and JPG are distinct contexts; Markdown sources are
preserved but are not submitted as duplicate text experiments.

Gitksan sources are `pdf1_brown` and `pdf2_rigsby`; other sources are `grammar`.
Variants are `original` and `cyrillic`. Selected PDF images are unchanged:
Gitksan PDF1=9, PDF2=6, Lezgi=9, Natugu=9, Tsez=9.

## Earlier Repair Campaign

Run from the GRAMMAMT directory. Listing does not call Slurm or a paid API.

```bash
python3 scripts/submit_jobs.py --family chain_gloss_v2 --dry-run
python3 scripts/submit_jobs.py --repairs --dry-run
python3 -m unittest discover -s tests -v
python3 runners/score_verified.py --targets configs/lezgi_missing_xcomet.json --dry-run
```

Every GPU job requests **one H100 only**. The original audited repair setup used FP32,
batch size one, greedy decoding, maximum 512 generated tokens, thinking disabled, with no
CPU offload or quantization. Preflight jobs request one hour. Generation requests
four hours, or ten for Tsez; scoring requests five hours.

```bash
python3 scripts/submit_jobs.py --preflight --submit
bash scripts/submit/chain_gloss_v2/submit_qwen3_natugu.sh
bash scripts/submit/chain_gloss_v2/submit_qwen35_natugu.sh
python3 scripts/submit_jobs.py --repairs --submit
sbatch scripts/scoring/run_lezgi_missing_xcomet.sh
```

Language wrappers also exist for Lezgi, Tsez, Lezgi Cyrillic, Gitksan PDF1 and
Gitksan PDF2. Ordinary curated jobs and corrected chain jobs are intentionally
separate. The earlier repair campaign comprised **70 chain conditions + 5 output
repairs**, not all 210 catalog entries. Historical successful Shot/ModelGloss
outputs are preserved. The newer matched family may require fresh controls
and material runs when the earlier prompts/settings do not form matched pairs.

Full submissions require a successful, current single-H100 FP32 preflight.
Unsupported attention kernels and CUDA OOM fail visibly; they do not trigger
another GPU, reduced images, or lower precision. The efficient SDPA backend is
explicitly selected; hardware success is not assumed. A failed preflight must
be investigated before its dependent experiments are submitted.

After generation completes, run `sbatch scripts/scoring/run_repair_xcomet.sh`.
The scorer validates complete test sets and computes XCOMET-XL only; it has no
COMET fallback. It preserves other metric values and skips scoring only when
the result hash, count, model and existing scores agree. Incomplete files are
reported as rejected, not silently included in complete-test tables.

## Reproducibility and Safety

- Test counts: Gitksan 37, Lezgi 87, Natugu 99, Tsez 445; support count 21.
- Generation writes each record durably. Resume checks source/reference order,
  context hashes, prompt/code fingerprints and settings. Invalid JSON or unknown
  provenance is not appended to.
- Gloss, raw attempts, final translation and invalid-output reasons are stored
  separately. Invalid predictions are never silently removed from the denominator.
- Historical result/metric contents and grammar image bytes are preserved.
- The old nine-row Tsez ModelGloss result is not a full-test result; see
  [the audit notes](docs/experiment_audit.md).
- `venv`, external DSS caches and the sibling `Database` are not relocated.
- API keys remain in the existing external environment file. Credentials,
  caches, bytecode and Slurm logs remain ignored by Git.
- `archive` is reference material, not a supported submission location. Migration
  utilities are one-time operations, not normal experiment commands.
- No GitHub push is performed as part of this migration.

Regenerate the active catalog and launchers with
`python3 scripts/generators/build_catalog.py`. This does not regenerate images,
alter existing predictions, or submit jobs.
