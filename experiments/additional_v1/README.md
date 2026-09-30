# Retained Additional Study

Read [EXPERIMENT_HANDOFF.md](EXPERIMENT_HANDOFF.md) for methods and limitations,
[publication/FINDINGS.md](publication/FINDINGS.md) for verified observations, and
[publication/condition_catalog.tsv](publication/condition_catalog.tsv) for outputs.

Delivered scope: 37 new Qwen3 conditions (6,049 records), fully lexical/XL/XXL
scored, plus completed historical Qwen3/Qwen3.5 A/B/C analyses. Failed new Qwen3.5
generation is excluded. All 192 retained truncations remain scored as empty translations.

Frozen original plan/code are preserved as provenance, including unsuccessful
planned conditions. They are **not** the delivered matrix. Historical submission
commands are not instructions to rerun the abandoned scope. Authoritative scope:
`publication/retained_manifest.json`. No automatic submission or push is performed.

Validate locally without generation or API requests:
```bash
venv/bin/python experiments/additional_v1/publication/finalize.py
```

The optional `--clean` operation is only for authorized failed-artifact cleanup.
Original primary-study scientific inputs/results remain unchanged. Reader-facing
documentation has been updated for this retained scope.
