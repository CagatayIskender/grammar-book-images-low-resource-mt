# Retained Supplementary Findings

Validated scope: 37 new Qwen3 conditions, 6,049 sentence-condition records; 192 truncated/empty outputs remain in all denominators.
Both XCOMET collections and lexical scores were checked against the retained prediction hashes and cohort masks.
Completed A/B/C analyses use historical results from both Qwen models; they are not new Qwen3.5 generations.

## Qwen3 Diagnostics

chrF++ differences on native cohorts (Lezgi valid84). F: 1024 minus 512 tokens. G: gold minus predicted gloss.
These are descriptive, not Holm-significant claims: the original F/G families remain incomplete.

| Language | F: budget difference | G: oracle difference | I: context difference, three seeds |
| --- | ---: | ---: | --- |
| Gitksan | +0.000 | -7.429 | +2.299, +0.218, +0.449 |
| Lezgi | +0.000 | -0.777 | +0.987, +0.872, +0.392 |
| Natugu | +0.000 | -2.931 | -0.829, -0.929, -0.264 |
| Tsez | +0.028 | +0.396 | +0.585, +0.506, +0.579 |

The larger budget changes little here; gold gloss is not a guaranteed performance upper bound.
The three selected context differences are positive for Gitksan, Lezgi and Tsez and negative for Natugu across the tested seeds.
This is useful diagnostic evidence, not proof of universal grammar benefits or general seed robustness.

## Completed Stored-Prediction Analyses

A: 46 of 72 metric tests have Holm-adjusted p < .05. This compares no-book methods, not grammar additions.
B: four of 360 tests have Holm-adjusted p < .05 (two contrasts evaluated under two overlapping masks).
Both significant context-baseline tests are negative chrF++ differences for Qwen3.5 Lezgi Cyrillic Chain-gloss cheat-sheet JPG.
The two significant format tests favor Cyrillic summary-tables JPG over TXT on BLEU; they are not baseline gains.
The 84/83 masks are sensitivity views, not independent replications. Exact estimates and p-values remain in reports/A_comparisons.tsv and reports/B_comparisons.tsv.

No XCOMET significance tests were performed. All completed learned-metric scores remain available, regardless of direction.
