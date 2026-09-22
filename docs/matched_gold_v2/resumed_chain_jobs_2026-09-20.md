# Resumed Chain Jobs

Four generation groups stopped at their configured 3,000-second checkpoint deadline, not because of CUDA OOM. They were resubmitted on 2026-09-20 with unchanged prompts, configurations, precision, and result destinations. The runner validates existing records, skips completed indices, and generates only missing records.

| Group | Previous job | Resume job | Missing records at submission | Slurm time limit | Runner deadline |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 Lezgi Chain | 5798003 | 5799942 | 532 | 3 hours | 165 minutes |
| Qwen3 Gitksan Chain | 5798018 | 5799943 | 5 | 1 hour | 50 minutes |
| Qwen3 Lezgi Chain | 5798021 | 5799944 | 660 | 3 hours | 165 minutes |
| Qwen3 Natugu Chain | 5798024 | 5799945 | 361 | 3 hours | 165 minutes |

Each job requests exactly one H100. All four jobs were confirmed pending after submission. Allocated time is an upper limit, not a predicted duration or a guarantee of completion.

## Scoring Dependencies

Job 5798036 (XCOMET XL) was temporarily held, updated, verified, and released. It now requires successful completion (`afterok`) of the remaining generation jobs:

`5798010, 5798013, 5798015, 5798016, 5798026, 5798027, 5798028, 5798029, 5798030, 5798031, 5798033, 5798034, 5799942, 5799943, 5799944, 5799945`.

Existing downstream dependencies remain unchanged:

- BLEU/chrF++ analysis 5798037 and XL analysis 5798172 require successful XL scoring 5798036.
- XXL scoring 5798139 requires successful XL scoring 5798036.
- XXL analysis 5798146 requires successful XXL scoring 5798139.

A failed generation job will block scoring rather than allow an incomplete campaign to advance. Scoring also retains its full-record validation. No existing prediction was deleted or regenerated during resubmission. Bash syntax validation passed for all four job scripts.
