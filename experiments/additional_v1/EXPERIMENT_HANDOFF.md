# Retained Supplementary Study: Thesis Handoff

This supplement is separate from the unchanged 351-condition primary study.
Use the retained scope below, not the original submission count, as the delivered
matrix. No further generation is planned in this delivery.

## Delivered Evidence

| Package | Retained scope | Interpretation |
| --- | --- | --- |
| A | Historical no-book Qwen3/Qwen3.5 comparisons; 72 BLEU/chrF++ tests | Method differences, not grammar-context gains |
| B | Historical Lezgi84/83 comparisons; 264 context-baseline and 96 format tests | Post-hoc masked inference, separate from primary statistics |
| C | Input-size tables and plots for both Qwen models | Logged counts and labelled reconstructions; not causal explanation |
| F | Four Qwen3 512/1024 no-book Chain-gloss pairs | Descriptive output-budget sensitivity |
| G | Four Qwen3 gold/predicted-gloss no-book pairs | Descriptive oracle diagnostic; not a guaranteed upper bound |
| I | Four Qwen3 context/baseline pairs, three seeds | Selected-subset stability, not full-matrix robustness |
| Metrics | BLEU, chrF++, XCOMET-XL/XXL for all 37 retained conditions | Empty predictions retained; no XCOMET significance tests |

There are **37 distinct new Qwen3 conditions and 6,049 sentence-condition records**.
Shared controls are counted once. **192 outputs are truncated/empty** and remain
in relevant denominators. Complete records do not mean every translation succeeded.
There are no missing records in the retained conditions.

The 37 planned new Qwen3.5 conditions produced no predictions because their jobs
activated the wrong environment. Their failed job files and empty artifacts were
removed at the user's request. Exclusion is operational, not based on unfavorable
scores. Completed historical Qwen3.5 analyses in A/B/C remain. The original plan
and frozen code are provenance, not an active job list.

## Cohorts and Protocol

| Language | Generated records per condition | Scoring views |
| --- | ---: | --- |
| Gitksan | 37 | Native37 |
| Lezgi | 87 | Valid84 and sensitivity83 |
| Natugu | 99 | Native99 |
| Tsez | 445 | Native445 and first99 |

Lezgi valid84 excludes zero-based indices 37, 62, 81 (invalid references).
Sensitivity83 also excludes index 11 (reference overlap with support). Tsez first99
is not a random sample or independent replication. Identical masks apply to both sides.

Original prompts, 21 gold-glossed supports and extraction are preserved. No-book
is not zero-shot. Only G deliberately supplies test gold glosses; other ModelGloss
conditions use predicted glosses. References are not added as query answers.
Historical overlap limitations remain.

Qwen3 uses one H100, FP32, thinking disabled, temperature 0.7, top-p 0.8, top-k 20,
min-p 0, presence penalty 1.5, repetition penalty 1. Output budget is 512 except
F's 1024-token intervention. No precision, resolution, image-count or prompt
fallback is introduced. Base seeds are 20260910, 20260925, 20261001; per-sentence
seed is base + twice original index. F/G use the first seed. Pinned revisions and
runtime metadata remain in configuration/results artifacts.

I compares grammar versus no-book for Gitksan Gloss-shot Brown cheat-sheet TXT,
Lezgi Chain-gloss cheat-sheet TXT, Natugu ModelGloss cheat-sheet JPG, and Tsez
Chain-gloss summary-text TXT. All seeds and outcome directions are retained.

## Inference Restrictions

A/B use 100,000 paired source-text cluster bootstrap draws, fixed seed 20260925,
95% pointwise percentile intervals and two-sided null-centered plus-one p-values.
Holm families remain A_methods (72), B_context (264), B_format (96), including
both metrics/models and declared cohorts within each family. chrF++ is primary;
BLEU supplementary. Report post-hoc status; overlapping masks are not independent
evidence. This differs from the original primary-study testing procedure.

F/G originally planned two-model families of 24 metric tests each. Those families
remain incomplete. Retained raw p-values and pointwise intervals are diagnostic;
**no Holm significance claim is made, and families were not reduced after results**.
I is descriptive. XL/XXL have no significance tests. Nonsignificance is not equivalence.

## Findings and Files

Start with [verified findings](publication/FINDINGS.md). These results have value
without showing universal improvement: a larger budget changes little, gold oracle
is not consistently better, and seed effects have mixed directions across languages.

| Path | Purpose |
| --- | --- |
| `publication/retained_manifest.json` | Authoritative retained scope and validation counts |
| `publication/condition_catalog.tsv` | All 37 conditions, completeness, failures and paths |
| `publication/artifact_hashes.json` | Prediction and XL/XXL artifact integrity |
| `reports/A_comparisons.tsv`, `reports/B_comparisons.tsv` | Completed stored-prediction inference |
| `reports/C_input_sizes_*.tsv` | Input-size diagnostics |
| `results/*.jsonl` | Raw responses and prompt evidence |
| `metrics/generation_scores.tsv` | Recomputed lexical scores |
| `metrics/generation_comparisons.tsv` | F/G/I comparisons, including unfavorable effects |
| `metrics/xcomet_xl/`, `metrics/xcomet_xxl/` | Verified segment scores |
| `reports/xcomet_xl_scores.tsv`, `reports/xcomet_xxl_scores.tsv` | Masked native-scale means |
| `configs/plan.json`, `configs/lock.json` | Original planned scope and frozen code provenance |

D/E provide automatic coverage/inventory only, not expert review. H/J/K and new
MTOB controls were not run. Gemini alignment drafts were abandoned, not executed.
Do not attribute findings to excluded packages. This handoff does not claim that
all originally submitted jobs succeeded.
