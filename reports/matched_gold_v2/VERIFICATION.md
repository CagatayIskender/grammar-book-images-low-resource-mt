# Final Handover Verification

Date: 22 September 2026. Scope: the corrected matched_gold_v2 handover.

- Finalizer completed successfully: 351 conditions, 40,731 records, 351 verified
  XL artifacts and 351 verified XXL artifacts. See audit.json for input hashes.
- Full-cohort BLEU/chrF++ were recomputed from recorded predictions and matched.
- Saved statistical input hashes and numerical TSV/JSON rows agreed.
- Four local unit tests passed: fixed nested reference masks; retention of failed
  predictions; rejection of overflowing figure labels; TSV zero/empty round trip.
- The publication check also passed all 18 tests selected by test_*v2*.py,
  including gold-support identity, target-gloss handling, prompt evidence,
  offline resume and prediction-bound XXL score reuse.
- Three repository-layout tests passed: one real metric location, one report
  location, and valid local links in reader-facing documentation. Redundant
  metric views and their recreation utility have been removed.
- Three metric-migration tests passed: idempotent/contained path resolution,
  unchanged hashes for all 702 relocated files, and rejection of unapproved
  runtime changes or altered original-code snapshots. All 53 current gold-v2
  generation/scoring shell scripts passed bash -n.
- After moving the report and grouping the 702 metric JSON files under two
  subdirectories of metrics/matched_gold_v2, original scientific artifact hashes
  were preserved, including all 1,523 prior audited input identities after path
  and original-code snapshot mapping. Derived catalogs now contain the new paths. Numerical tables
  and values are unchanged. See docs/metric_layout_migration.json for old/new
  paths and original/approved runtime code identities.
- The recommended score table has 702 unique condition/cohort rows. All Lezgi
  reporting rows use 84 evaluated references and a descriptive-only inference label.
- Local report links are checked after directory changes. Methodological
  corrections remain documented in THESIS_GUIDE.md.
- All eight JPGs passed dimension and nonblank-pixel checks. Figure generation
  checks the bounds of every text label and raises an error on overflow.
- The overview, detailed matrix and a representative appendix page were visually
  inspected. The appendix PDF has six pages. This is not a manual inspection of
  every cell's translation quality or an independent OCR audit of book images.
- Python tests and shell syntax checks passed. Whitespace checking treats TSV
  CRLF line endings as intentional and excludes byte-preserved archived source
  snapshots, whose original trailing blank lines must not be normalized.
- Scheduler verification showed no remaining user jobs. Scoring/analysis jobs
  5798036, 5798037, 5798139, 5798146 and 5798172 completed with exit code 0:0.

Frozen prompts, recorded predictions and original metric bytes were not changed.
Active generation/scoring code received path-only I/O and validation adapters;
original code snapshots remain available. Existing statistical numbers were preserved; their Markdown reports
received reference-quality qualifications. No new Slurm job, paid API request,
GitHub push or external publication was performed by the finalizer itself.
Repository publication is a separate user-authorized step after credential checks.

Lezgi sensitivities are post hoc descriptive checks, not new significance tests.
Full historical-file hash revalidation is outside this corrected-input audit;
the previous historical artifacts were not edited by this finalization.
