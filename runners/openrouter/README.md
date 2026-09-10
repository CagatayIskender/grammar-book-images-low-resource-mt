# OpenRouter Runners

See the [project guide](../../README.md) for the current layout. No paid API
experiments are started by the Qwen repair campaign.

experiment_sources.json selects frozen materials from materials/curated_v1.
run_openrouter_suite.py writes matching results/metrics trees organized by
family, model, language, PDF source and variant. Smoke outputs are separate.
The existing external API-key environment file is unchanged; do not put keys
in source code or configuration JSON.

From the project root, validate inputs without sending API requests:

```bash
python3 runners/openrouter/run_openrouter_suite.py --source natugu --validate_only
python3 runners/openrouter/run_openrouter_suite.py --source tsez --test_n 99 --validate_only
```

Existing Gemini/Luna launchers remain in scripts/jobs/openrouter and submit
wrappers in scripts/submit/curated_v1. They retain the 99-sentence Tsez limit.
score_openrouter_xcomet.py uses the separate verified XCOMET scorer and the
organized output tree; smoke files are not included by default.
