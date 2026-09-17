# Local Validation

Validated on 2026-09-17 before the final CPU evaluation submission.

| Check | Result |
|---|---|
| Existing suite plus new closure tests | 32 tests passed |
| Final closure-only regression rerun | 18 tests passed |
| Existing extraction/retrieval/prompt/thinking validator | Passed with local checksum-pinned GPT-2 data |
| Bash syntax validation | All 83 MTOB job/submit shell files passed |
| Full Qwen audit | 30 conditions, 4,230 records, no identity/index/source/reference errors |
| Original input/result/historical metric hashes | Unchanged after audit |
| Prompt reconstruction | Exact Appendix C prompt/context matches; source occurs twice |
| Chunk reconstruction | Exact GPT-2 512-token windows, 256-token overlap, page/character mapping |
| Gl reconstruction | Original selected page order and actual token lengths match manifests |
| Metric/bootstrap alignment | Unsmooth 13a BLEU and plain chrF agree on controlled fixtures |
| Failure denominator | Refusals/empty/reasoning records retained as blank predictions |
| Parser | Multi-sentence text retained; ambiguous candidates preserved; reference-blind rules |
| Checkpoint reuse | Matching bindings reused; altered bindings/errors rejected |
| Degenerate significance | Identical systems receive conservative p=1; original library p retained |
| Runtime write boundary | Attempted write outside evaluation_v1 rejected |
| Runtime network boundary | DNS/network attempt rejected |
| Git whitespace validation | Passed |

Bootstrap unit fixtures use 20 resamples to test mechanics, determinism and
serialization. These are not production significance results. The production
analysis is fixed at 100,000 resamples and 120 planned tests. Its authoritative
completion status is in `analysis/results.json` and `CLOSURE_REPORT.md`.

No translation generation or inference API request was made. Only public,
checksum-pinned GPT-2 tokenizer data were provisioned during setup because the
current node's temporary tokenizer cache was missing. The closure itself is offline.

Initial evaluation jobs 5795278 and 5795279 were cancelled before the final
submission to finish a conservative parser-boundary regression check. They are
not the active evaluation jobs. Final IDs are recorded in `submissions/`.

This validation does not independently certify the original linguistic curation,
unrecorded historic runtime settings, or the absence of test examples in grammar
books. It does not claim to recover overwritten earlier-round responses.

## Completed Scoring Check

Final score job **5795286** completed successfully in **23 seconds**. Final
analysis job **5795287** started with its `afterok` dependency satisfied.
Both jobs request one CPU, 24 GB RAM and a two-hour limit; neither requests a GPU.

- All 30 conditions have complete scores in both views.
- Both views include exactly 4,230 records overall.
- Natugu Qwen3.5 Ge and Gs each have 99 records in both views.
- All 60 view-level prediction/reference hashes and metric configurations match the bootstrap inputs.
- Reproducing the historical status-ok evaluation gives zero numerical deviation for BLEU, chrF, ROUGE1/2/L and CharacTER across all 30 conditions.
- The 139 protected original files still match the frozen input hashes.

Significance completion is reported by `analysis/results.json`; this scoring
validation does not assert that the 120 production tests have already finished.
