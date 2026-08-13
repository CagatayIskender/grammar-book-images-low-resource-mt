# Run Scripts Layout

Use this folder as the script index.

- `submit/`: submit helpers for Slurm jobs.
- `qwen3/`: Qwen 3 generation jobs, grouped by language and PDF.
- `qwen35/`: Qwen 3.5 generation and scoring jobs.
- `qwen35/scoring/`: XCOMET scoring jobs for Qwen 3.5 outputs.
- `shared/`: shared environment setup used by run scripts.
- `generators/`: scripts that regenerate experiment run scripts.
- `legacy/baselines/`: older non-context or non-Qwen-baseline jobs.

Common Qwen 3.5 submit groups:

```bash
bash "Run Scripts/submit/submit_qwen35_gitksan_pdf1.sh"
bash "Run Scripts/submit/submit_qwen35_gitksan_pdf2.sh"
bash "Run Scripts/submit/submit_qwen35_lezgi.sh"
bash "Run Scripts/submit/submit_qwen35_natugu.sh"
```

After all Qwen 3.5 generation jobs finish:

```bash
sbatch "Run Scripts/qwen35/scoring/run_qwen35_score_xcomet_all.sh"
```
