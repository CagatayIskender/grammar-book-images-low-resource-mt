# MTOB versus GRAMMAMT Baselines

This is a cross-protocol translation-performance comparison, NOT an isolated estimate of the effect of grammar.

## Completion

Cohorts verified: 30 MTOB systems and 24 unique matched_v1 baseline systems; 90 paired comparisons.
Significance: 360/360 verified tests. State: complete. No significance claim before all tests complete.
No new generation, API calls, GPU allocation, historical output replacement, or changes to the 351-condition primary matrix.

## Common Evaluation

Every source sentence and reference matches exactly, by index, within each paired comparison. No successful-only subsets.
Gitksan PDF1/PDF2 share the same language baseline and test sentences; they are not independent datasets.
Raw view means the translation response after removal of required output markers/gloss, not the full chain-gloss response.
Extracted view additionally uses the frozen, reference-blind MTOB explanation parser. Multi-sentence translations are preserved.
BLEU and chrF are displayed on 0-100 scales. BLEU: 13a, no smoothing, no effective order; chrF: character order 6, word order 0, beta 2.
Both use paper punctuation cleanup. ROUGE uses local macro F1 (0-1); CharacTER is word-shift-aware, hypothesis-normalized and capped at 1, not standard CER.
Percentage = 100*(MTOB-baseline)/baseline; undefined for zero baselines. Near-zero baselines can produce misleadingly large percentages.
Negative CharacTER changes indicate improvement. Non-significance is not equivalence.
All 360 planned BLEU/chrF tests across both views share one Holm family; 100,000 paired bootstrap resamples, seed 20260917.
Counts below are descriptive comparisons, not independent trials, pooled corpus scores, or a causal analysis.

## Descriptive Comparison Counts

| Model | Baseline method | View | Metric | Higher / lower / equal | Significant higher / lower |
|---|---|---|---|---|---|
| qwen3 | shot | raw | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen3 | shot | raw | chrF | 0 / 15 / 0 | 0 / 10 |
| qwen3 | shot | extracted | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen3 | shot | extracted | chrF | 0 / 15 / 0 | 0 / 4 |
| qwen3 | chain_gloss | raw | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen3 | chain_gloss | raw | chrF | 0 / 15 / 0 | 0 / 14 |
| qwen3 | chain_gloss | extracted | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen3 | chain_gloss | extracted | chrF | 0 / 15 / 0 | 0 / 11 |
| qwen3 | modelgloss | raw | BLEU | 0 / 15 / 0 | 0 / 12 |
| qwen3 | modelgloss | raw | chrF | 0 / 15 / 0 | 0 / 15 |
| qwen3 | modelgloss | extracted | BLEU | 0 / 15 / 0 | 0 / 12 |
| qwen3 | modelgloss | extracted | chrF | 0 / 15 / 0 | 0 / 15 |
| qwen35 | shot | raw | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen35 | shot | raw | chrF | 0 / 15 / 0 | 0 / 11 |
| qwen35 | shot | extracted | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen35 | shot | extracted | chrF | 0 / 15 / 0 | 0 / 8 |
| qwen35 | chain_gloss | raw | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen35 | chain_gloss | raw | chrF | 0 / 15 / 0 | 0 / 12 |
| qwen35 | chain_gloss | extracted | BLEU | 0 / 15 / 0 | 0 / 3 |
| qwen35 | chain_gloss | extracted | chrF | 0 / 15 / 0 | 0 / 9 |
| qwen35 | modelgloss | raw | BLEU | 0 / 15 / 0 | 0 / 15 |
| qwen35 | modelgloss | raw | chrF | 0 / 15 / 0 | 0 / 15 |
| qwen35 | modelgloss | extracted | BLEU | 0 / 15 / 0 | 0 / 15 |
| qwen35 | modelgloss | extracted | chrF | 0 / 15 / 0 | 0 / 15 |

## Absolute Scores: Raw View

Each cell is BLEU / chrF, both on a 0-100 scale. Baseline columns contain no grammar book.

| Model | Source | N | Shot | Chain-gloss | ModelGloss | Ge | Gs | Gl |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | gitksan_pdf1 | 37 | 3.06 / 24.96 | 2.39 / 27.00 | 14.17 / 48.74 | 0.00 / 17.39 | 0.00 / 17.20 | 0.00 / 19.84 |
| qwen3 | gitksan_pdf2 | 37 | 3.06 / 24.96 | 2.39 / 27.00 | 14.17 / 48.74 | 0.00 / 15.80 | 0.28 / 14.69 | 0.00 / 20.23 |
| qwen3 | natugu | 99 | 2.07 / 20.15 | 2.62 / 20.78 | 10.61 / 37.88 | 0.47 / 14.35 | 1.07 / 15.62 | 1.21 / 15.08 |
| qwen3 | lezgi | 87 | 6.26 / 22.93 | 6.61 / 23.97 | 12.01 / 38.03 | 0.18 / 11.84 | 0.19 / 11.94 | 0.53 / 16.98 |
| qwen3 | tsez | 445 | 0.99 / 20.03 | 0.79 / 20.13 | 8.67 / 40.62 | 0.09 / 14.70 | 0.00 / 13.38 | 0.00 / 17.89 |
| qwen35 | gitksan_pdf1 | 37 | 2.38 / 26.12 | 2.94 / 27.36 | 12.05 / 46.55 | 0.70 / 21.73 | 0.58 / 19.57 | 0.58 / 20.00 |
| qwen35 | gitksan_pdf2 | 37 | 2.38 / 26.12 | 2.94 / 27.36 | 12.05 / 46.55 | 0.61 / 18.58 | 0.39 / 17.62 | 0.28 / 16.73 |
| qwen35 | natugu | 99 | 2.56 / 21.46 | 2.61 / 21.19 | 10.04 / 39.26 | 0.61 / 15.45 | 1.86 / 18.57 | 1.15 / 15.72 |
| qwen35 | lezgi | 87 | 4.97 / 28.48 | 7.03 / 29.53 | 8.90 / 39.16 | 1.24 / 22.19 | 1.91 / 23.08 | 0.42 / 14.75 |
| qwen35 | tsez | 445 | 1.29 / 20.58 | 1.29 / 21.02 | 10.13 / 40.37 | 0.00 / 18.17 | 0.00 / 17.93 | 0.11 / 14.95 |

## Absolute Scores: Extracted View

Each cell is BLEU / chrF, both on a 0-100 scale. Baseline columns contain no grammar book.

| Model | Source | N | Shot | Chain-gloss | ModelGloss | Ge | Gs | Gl |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | gitksan_pdf1 | 37 | 3.06 / 24.96 | 2.39 / 27.00 | 14.17 / 48.74 | 0.00 / 18.59 | 0.00 / 19.65 | 0.00 / 20.10 |
| qwen3 | gitksan_pdf2 | 37 | 3.06 / 24.96 | 2.39 / 27.00 | 14.17 / 48.74 | 0.00 / 17.62 | 0.36 / 16.63 | 0.00 / 20.23 |
| qwen3 | natugu | 99 | 2.07 / 20.15 | 2.62 / 20.78 | 10.61 / 37.88 | 0.92 / 17.40 | 1.94 / 19.55 | 1.31 / 15.79 |
| qwen3 | lezgi | 87 | 6.26 / 22.93 | 6.61 / 23.97 | 12.01 / 38.03 | 0.25 / 13.76 | 0.26 / 13.83 | 0.56 / 17.22 |
| qwen3 | tsez | 445 | 0.99 / 20.03 | 0.79 / 20.13 | 8.67 / 40.62 | 0.10 / 15.46 | 0.00 / 16.67 | 0.00 / 17.89 |
| qwen35 | gitksan_pdf1 | 37 | 2.38 / 26.12 | 2.94 / 27.36 | 12.05 / 46.55 | 0.78 / 22.74 | 0.83 / 22.59 | 0.88 / 23.54 |
| qwen35 | gitksan_pdf2 | 37 | 2.38 / 26.12 | 2.94 / 27.36 | 12.05 / 46.55 | 0.77 / 20.55 | 0.48 / 19.59 | 0.36 / 18.41 |
| qwen35 | natugu | 99 | 2.56 / 21.46 | 2.61 / 21.19 | 10.04 / 39.26 | 0.78 / 16.83 | 2.07 / 19.41 | 2.38 / 19.84 |
| qwen35 | lezgi | 87 | 4.97 / 28.48 | 7.03 / 29.53 | 8.90 / 39.16 | 1.39 / 22.96 | 2.08 / 23.63 | 0.84 / 20.04 |
| qwen35 | tsez | 445 | 1.29 / 20.58 | 1.29 / 21.02 | 10.13 / 40.37 | 0.00 / 18.43 | 0.00 / 18.83 | 0.17 / 17.41 |

## Interpretation Limits

- Matched baselines use method-specific GRAMMAMT support/examples or gloss steps; MTOB uses the Appendix C direct translation prompt with grammar passages.
- MTOB: temperature 0.05, output cap 256. Matched_v1: temperature 0.7, output cap 512; seeds and other sampling controls differ.
- MTOB can use refusal/reasoning retries; matched_v1 retains first semantic attempts. Four Qwen3.5 MTOB records were rerun; earlier responses are unavailable.
- MTOB truncation metadata is unavailable. Matched_v1 records it; absence of MTOB metadata is not evidence of no truncation.
- Five Qwen3 chain baseline records have no usable translation; they remain blank predictions. Both views retain all expected sentences.
- Gl contexts differ in actual length; not all are 100K tokens. Different source grammars and retrieval strategies expose different information.
- Common evaluation makes output quality comparable; it does not equalize prompts, computation, information budgets or retry policies.
- Higher scores support a claim about these evaluated system configurations only. They do not show that adding grammar alone caused improvement.
- This analysis does not generate the cancelled MTOB no-book baselines, does not include API models, and does not compare MTOB with every grammar-augmented matched condition.

## Artifacts

- `comparisons.tsv`: per-condition scores, absolute/percentage differences and adjusted significance.
- `summary.tsv`: descriptive direction counts by model, baseline method and view.
- `cohort_audit.tsv`, `planned_comparisons.tsv`, `audit.json`: matching, exclusions (none), failures and extraction diagnostics.
- `records/`: original response, exact selected span, extraction rules and both evaluation views.
- `scores/`, `tests/`, `significance.tsv`: common metrics and reproducible paired tests.
- `provenance.json`: input/config/normalized-view hashes, parser implementation and evaluator settings.
