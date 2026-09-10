# MTOB Grammar Experiments

An isolated implementation of the grammar-only conditions from Tanzer et al.
(ICLR 2024): embedding retrieval (`Ge`), longest-common-substring retrieval
(`Gs`), and manually curated long grammar context (`Gl`).

The suite reads the existing grammar PDFs and SIGMORPHON data but writes only
below this directory. It does not import the existing experiment runners.

## Setup and preparation

```bash
bash setup_env.sh
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python extract_grammar.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python validate_extraction.py
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python build_chunks.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python build_embedding_index.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python build_retrieval_manifests.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python curate_gl.py --all
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python snapshot_prompts.py
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python generate_jobs.py
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python validate_experiments.py
```

The cached chat templates can be checked in their matching environments with:

```bash
source ../venv/bin/activate
PYTHONPATH="$PWD/vendor:$PWD" TRANSFORMERS_OFFLINE=1 python verify_qwen_thinking.py --model qwen3
source /dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/envs/grammamt-qwen35/bin/activate
PYTHONPATH="$PWD/vendor:$PWD" TRANSFORMERS_OFFLINE=1 python verify_qwen_thinking.py --model qwen35
```

`validate_extraction.py` prints representative page starts and extraction
warnings. Review that report before treating the generated plaintext as audited.

## Inspect and run

```bash
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python list_experiments.py
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python run_experiment.py \
  --model qwen35 --source natugu --condition gs --smoke
```

The full matrix has five grammar sources, three conditions, and four models:
60 experiments. Submit scripts are grouped by model and language under
`submit/`; none are submitted automatically.

Examples:

```bash
# Show all 60 experiment states.
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python list_experiments.py

# Submit Qwen 3 for Gitksan PDF 1 (Ge, Gs, Gl).
bash submit/submit_qwen3_gitksan_pdf1.sh

# Recompute metrics after a completed experiment or for all available results.
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python evaluate_results.py \
  --model qwen3 --source gitksan_pdf1 --condition ge
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python evaluate_results.py --all

# Confirm source PDFs and test sets still match the implementation baseline.
PYTHONPATH="$PWD/vendor:$PWD" ../venv/bin/python capture_input_hashes.py --verify
```

Each model/source submit script submits exactly three independent jobs. Gitksan
PDF 1 and PDF 2 have separate scripts. Results are resumable JSONL files and
metrics are written only under this suite.

## Faithfulness notes

- Prompts reproduce Appendix C for language-to-English translation.
- `Ge` and `Gs` use 512 GPT-2-token chunks, 256-token overlap, and two passages.
- `Gs` uses raw longest-common-substring length, matching the released code.
- `Ge` uses `sentence-transformers/all-mpnet-base-v2` cosine similarity.
- `Gl` uses manually listed page ranges and the paper's full-book delimiters.
- `Gm` is intentionally absent because its paper definition requires a complete
  authoritative bilingual wordlist.
- Thinking is disabled through backend controls without changing the prompt.
- Evaluation follows the paper's punctuation cleaning and reports chrF, BLEU,
  ROUGE, and character error rate. XCOMET is intentionally absent.
