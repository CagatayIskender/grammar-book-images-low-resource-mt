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
- The recommended score table has 702 unique condition/cohort rows. All Lezgi
  reporting rows use 84 evaluated references and a descriptive-only inference label.
- After moving the handover to thesis/, all 74 local links checked across the
  root README, thesis guide, handover documents and corrected-study documents
  resolved. The missing supervisor summary was supplied before publication.
- All eight JPGs passed dimension and nonblank-pixel checks. Figure generation
  checks the bounds of every text label and raises an error on overflow.
- The overview, detailed matrix and a representative appendix page were visually
  inspected. The appendix PDF has six pages. This is not a manual inspection of
  every cell's translation quality or an independent OCR audit of book images.
- Python syntax checks and git diff whitespace validation passed during preparation.
- Scheduler verification showed no remaining user jobs. Scoring/analysis jobs
  5798036, 5798037, 5798139, 5798146 and 5798172 completed with exit code 0:0.

No generation code, frozen prompt, recorded prediction or original metric was
changed. Existing statistical numbers were preserved; their Markdown reports
received reference-quality qualifications. No new Slurm job, paid API request,
GitHub push or external publication was performed by the finalizer itself.
Repository publication is a separate user-authorized step after credential checks.

Lezgi sensitivities are post hoc descriptive checks, not new significance tests.
Full historical-file hash revalidation is outside this corrected-input audit;
the previous historical artifacts were not edited by this finalization.
