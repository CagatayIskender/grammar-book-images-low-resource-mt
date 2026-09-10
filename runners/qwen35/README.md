# Qwen 3.5 Runners

See the [project guide](../../README.md) and [experiment catalog](../../docs/experiment_catalog.tsv).

Current curated and chain-gloss jobs use ../run_audited_context.py with an explicit
configuration in ../../configs/experiments. They request one H100, FP32, no
offload, no quantization and disabled thinking.

run_grammamt_Qwen35_context.py retains historical prompt builders for compatible
Shot/ModelGloss repair prompts and legacy experiments. Its historical Chain
builder is NOT the corrected chain_gloss_v2 condition. Do not use historical
duplicate Shot/Chain scores as independent chain-gloss evidence.

score_qwen35_xcomet_from_jsonl.py is a compatibility CLI for ../score_verified.py.
It searches the reorganized results tree and requires complete, source-matched
test records. XCOMET runs in a separate base-environment job.

The external Qwen3.5 environment and DSS model cache remain in their existing
locations. scripts/shared/qwen35_job_env.sh at the project root activates them.
