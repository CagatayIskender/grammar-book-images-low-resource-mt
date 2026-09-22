# Original Code Before Metric Relocation

These five files are byte-for-byte snapshots of production code before the
September 2026 metric-directory cleanup. They are provenance evidence, not
supported entry points; do not execute them from this archive.

Active code now resolves saved metric paths into:

```text
metrics/matched_gold_v2/lexical_and_xcomet_xl/
metrics/matched_gold_v2/xcomet_xxl/
```

The original configuration fingerprints, predictions, metric JSON bytes and
statistical result files were not rewritten. The migration manifest at
docs/metric_layout_migration.json records all 702 old/new metric paths and
hashes, plus original and approved runtime code hashes. Validation checks both
the original snapshot and the exact approved adapter version; arbitrary code
edits are not accepted as compatible. The XXL scorer accepts the original
scorer identity only for this explicitly recorded path-only migration; all
other prediction, checkpoint, configuration and evaluation settings must match.

Generation prompts, decoding, parsing, model calls and numerical scoring logic
are unchanged. Changes to active code concern metric I/O and migration-aware
validation only. Use the active runners in runners/ for reproduction.
