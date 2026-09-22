# Actual Model Input Audit

Snapshot: 2026-09-20T19:29:55.721612+00:00
Planned prompts checked for test-gold/reference noninterference: 40731.
Recorded messages: {'gemini25flashlite': 8733, 'qwen35': 9510, 'qwen3': 5434}; API request-hash mismatches: 0.
All audited records pass reconstruction of gold-support messages; supports contain 21 nonempty gold glosses.
ModelGloss test slots exactly match the corresponding external unsegmented pred field, not gold.
No claim is made that test-reference strings or gold-gloss strings are absent everywhere in context.
Literal-match events: 1014. Placeholder references: 3.
Lezgi test index 11 has its reference as a substring of a longer support translation (support index 19).
Some external predictions equal gold exactly; this is not evidence of reading the gold field.
See report.json for every match, source and snapshot hash. Images have not been OCR-audited here.
No jobs, prompts, references, support sets, or predictions were changed by this audit.
