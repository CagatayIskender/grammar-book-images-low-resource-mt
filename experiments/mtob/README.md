# MTOB Grammar Experiments

An isolated implementation of the grammar-only conditions from Tanzer et al.
(ICLR 2024): embedding retrieval (`Ge`), longest-common-substring retrieval
(`Gs`), and manually curated long grammar context (`Gl`).

The suite reads the existing grammar PDFs and SIGMORPHON data but writes only
below this directory. It does not import the existing experiment runners.

## Qwen supplementary closure

The current handover uses [evaluation_v1](evaluation_v1/README.md) for the 30
existing Qwen conditions. See the [closure report](evaluation_v1/CLOSURE_REPORT.md)
and [condition matrix](evaluation_v1/condition_matrix.tsv). The closure preserves
historical predictions/metrics and adds raw/extracted, full-denominator evaluation.
It is separate from the main 351-condition matched_v1 study. No new generation is
needed for this closure. The preparation and generation commands below are for
reproduction only, not instructions to rerun completed experiments.

## Setup and preparation

```bash
bash setup_env.sh
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python extract_grammar.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python validate_extraction.py
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python build_chunks.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python build_embedding_index.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python build_retrieval_manifests.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python curate_gl.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python snapshot_prompts.py
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python generate_jobs.py
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python validate_experiments.py
```

The cached chat templates can be checked in their matching environments with:

```bash
source ../../venv/bin/activate
PYTHONPATH="$PWD/vendor:$PWD" TRANSFORMERS_OFFLINE=1 python verify_qwen_thinking.py --model qwen3
source /dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/envs/grammamt-qwen35/bin/activate
PYTHONPATH="$PWD/vendor:$PWD" TRANSFORMERS_OFFLINE=1 python verify_qwen_thinking.py --model qwen35
```

`validate_extraction.py` prints representative page starts and extraction
warnings. Review that report before treating the generated plaintext as audited.

## Inspect and run

```bash
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python list_experiments.py
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python run_experiment.py \
  --model qwen35 --source natugu --condition gs --smoke
```

The full matrix has five grammar sources, three conditions, and four models:
60 experiments. Submit scripts are grouped by model and language under
`submit/`; none are submitted automatically.

Examples:

```bash
# Show all 60 experiment states.
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python list_experiments.py --all-models

# Submit Qwen 3 for Gitksan PDF 1 (Ge, Gs, Gl).
bash submit/submit_qwen3_gitksan_pdf1.sh

# Audit existing Qwen results without modifying historical metrics.
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python finalize_results.py --stage audit
# Submit CPU scoring followed by dependent significance analysis.
bash submit/evaluation/submit_evaluation_v1.sh

# Confirm source PDFs and test sets still match the implementation baseline.
PYTHONPATH="$PWD/vendor:$PWD" ../../venv/bin/python capture_input_hashes.py --verify
```

Each model/source submit script submits exactly three independent jobs. Gitksan
PDF 1 and PDF 2 have separate scripts. Results are resumable JSONL files and
metrics are written only under this suite.

## Faithfulness notes

For comparison with existing GRAMMAMT no-book baselines, use the separate
[cross-protocol evaluation](grammamt_comparison_v1/README.md) and its
[current report](grammamt_comparison_v1/COMPARISON_REPORT.md). It recomputes both
setups with common scoring and exact sentence pairing. This is a comparison of
complete configurations, not a controlled grammar-addition ablation. It does not
generate any new translations or restart the cancelled baseline extension.

- Prompts reproduce Appendix C for language-to-English translation.
- `Ge` and `Gs` use 512 GPT-2-token chunks, 256-token overlap, and two passages.
- `Gs` uses raw longest-common-substring length, matching the released code.
- `Ge` uses `sentence-transformers/all-mpnet-base-v2` cosine similarity.
- `Gl` uses manually listed page ranges and the paper's full-book delimiters.
- `Gm` is intentionally absent because its paper definition requires a complete
  authoritative bilingual wordlist.
- Thinking is disabled through backend controls without changing the prompt.
- Evaluation uses the paper's punctuation cleaning and reports chrF, BLEU,
  the existing local ROUGE implementation, and CharacTER (not standard character
  error rate). XCOMET is intentionally absent. The legacy `evaluate_results.py`
  filters `status=ok` and overwrites old metrics; use `finalize_results.py` for
  the supplementary closure and its full-denominator, two-view evaluation.
