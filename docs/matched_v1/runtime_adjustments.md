# Runtime adjustments (2026-09-15)

These are conservative scheduling estimates, not measured matched-v1 runtimes.
Prompts, datasets, decoding, image inputs and FP32 are unchanged. One GPU per job.

| Pending work | Previous limit | New limit | Evidence and buffer |
| --- | --- | --- | --- |
| Four non-Tsez chain continuations | 10 h | 1 h | Only 12, 37, 98 or 99 missing rows; existing rows retained |
| Gitksan Shot/ModelGloss, four groups | 10 h | 4 h | 481 rows/group; historical 37-row conditions about 3-11 min; generous allowance for changed method |
| Lezgi/Natugu Shot/ModelGloss, eight groups | 10 h | 6 h | 693-1044 rows/group; historical 87/99-row conditions about 5-38 min; method and context overhead buffer |
| Tsez summary-text-only Shot/ModelGloss, four groups | 10 h | 3 h | 445 rows/group; historical summary-text chain about 36-40 min; over fourfold allowance |
| Other Tsez batches, twelve groups | 10 h | 10 h | Up to 1335 rows, multiple contexts; insufficient evidence to shorten safely |

Historical Slurm evidence (`sacct`, same user's jobs):
- 5784395: Gitksan Qwen3 selected-page chain, 00:10:27, COMPLETED.
- 5784397: Lezgi Qwen3 selected-page chain, 00:16:39, COMPLETED.
- 5788198 / 5788200: Natugu Qwen3 image chain, 00:23:57 / 00:25:14, COMPLETED.
- 5780307: Natugu Qwen3 selected-page ModelGloss, 00:26:27, COMPLETED.
- 5784400 / 5784394: Tsez summary-text chain Qwen3/Qwen35, 00:40:22 / 00:36:02, COMPLETED.
- 5784401 / 5780308 / 5780309: Tsez selected-page Shot/ModelGloss, about 1 h 45 min to 1 h 58 min, COMPLETED.
- 5784398 / 5784399 ran 00:37:21 / 02:43:51 but ended FAILED; these are not successful-run proofs and are not used to justify aggressive cuts.

Historical per-condition status files also report runtime_seconds. These can describe
only a resumed invocation, so they must not be blindly divided by final row counts.
Different methods and changed prompts make these timing proxies, not exact forecasts.
Group limits allow substantial overhead; heavy Tsez multi-condition batches stay at 10 h.

Worker deadlines are 10 minutes below a 1-hour allocation, otherwise 30 minutes below
the allocation. Partial outputs remain resumable; deadline exit 75 requires a later
continuation, not a prompt change. Completion is not guaranteed by these estimates.

The generator uses validated existing row counts for short chain continuations; a
fresh full chain group will receive the longer language-level limit.
No API jobs are resubmitted. XCOMET (14 h) and the two CPU analyses (10 h each)
retain their limits: the new scoring volume / 100000-resample workload does not have
a sufficiently comparable measured completed run to support a safe reduction here.

Replacement IDs and exact scoring dependencies are recorded in
`walltime_resubmissions.json`. Scoring jobs are held during replacement, then
rebound to the current generation IDs before release. Shorter limits may help
scheduling, but cancellation/resubmission resets submission age and an earlier
start is not guaranteed.
