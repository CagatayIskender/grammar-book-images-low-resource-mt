# Leveraging Grammar-Book Images in Vision-Language Models for Low-Resource Machine Translation

Code, saved model outputs, scores and reports for the master's thesis by
**Cagatay Iskenderoglu** (Technical University of Munich, Informatics; supervisor:
Shu Okabe, Ph.D.).

**Thesis:** [thesis/Cagatay_Iskenderoglu_Masters_Thesis.pdf](thesis/Cagatay_Iskenderoglu_Masters_Thesis.pdf)

## Overview

The thesis asks what selected grammar-book context adds to a translation prompt
that already contains glossed examples. It translates sentences from **Gitksan,
Lezgi, Natugu and Tsez** into English with three vision-language models, using
three prompting methods adapted from GRAMMAMT (Gloss-shot, Chain-gloss and
ModelGloss), each with 21 gold-glossed training examples. Across **351
conditions**, each grammar package (text summaries, summary tables, cheat sheets
and selected original grammar pages; tables and cheat sheets as both text and
images) is compared with a matched prompt that has no grammar context.

Main findings:

1. Without grammar context, ModelGloss has the highest BLEU and chrF++ in every
   model–language group.
2. Adding grammar context gives mixed results: in the corrected tests, one
   contrast improves and ten decrease, all ten with image packages.
3. All nine significant text-vs-image differences favour the text version.

The thesis abstract and results chapter give the exact scope of these claims.

## Repository at a Glance

| Need | Open |
| --- | --- |
| Methods, claims and limitations | [THESIS_GUIDE.md](THESIS_GUIDE.md) |
| Final tables, figures and completion audit | [reports/matched_gold_v2/](reports/matched_gold_v2/README.md) |
| Recommended score table | [thesis_scores.tsv](reports/matched_gold_v2/thesis_scores.tsv) |
| Significance results | [significance_summary.tsv](reports/matched_gold_v2/significance_summary.tsv) |
| Every condition with its config, outputs and scores | [condition_catalog.tsv](reports/matched_gold_v2/condition_catalog.tsv) |
| Experiment matrix at a glance | [experiment_matrix_overview.jpg](reports/matched_gold_v2/figures/experiment_matrix_overview.jpg) |
| Folder and version explanations | [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) |
| Supplementary Qwen diagnostics | [experiments/additional_v1/](experiments/additional_v1/EXPERIMENT_HANDOFF.md) |

| Folder | Contents |
| --- | --- |
| `thesis/` | The thesis PDF |
| `reports/` | Reader-facing tables, figures and audits |
| `configs/` | Frozen run settings, inputs and fingerprints |
| `results/` | Saved model responses (JSONL) |
| `metrics/` | Per-condition scores (BLEU, chrF++, XCOMET-XL, XCOMET-XXL) |
| `materials/`, `inputs/` | Grammar packages given to the models; gold support examples |
| `runners/`, `scripts/` | Generation, scoring, analysis and report code |
| `tests/` | Verification tests |
| `docs/` | Protocol details, statistical analyses and provenance |
| `experiments/mtob/` | Separate supplementary MTOB study, not part of the 351 conditions |
| `archive/` | Superseded or historical material, kept for provenance only |

**Which experiment is the thesis?** `matched_gold_v2` is the primary study.
`matched_v1`, `curated_v1`, `chain_gloss_v2`, `baseline`, `legacy` and `smoke` are
earlier or pilot experiments; do not pool them with the primary results. Version
numbers are local to each experiment family, so `chain_gloss_v2` is not
`matched_gold_v2`.

## Resources

### Models

| Role | Model | Link |
| --- | --- | --- |
| Translation | Qwen3-VL-8B-Instruct | [huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Translation | Qwen3.5-9B | [huggingface.co/Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Translation | Gemini 2.5 Flash Lite, via OpenRouter | [Google docs](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite) · [OpenRouter](https://openrouter.ai/google/gemini-2.5-flash-lite) |
| Predicted glosses for ModelGloss | GlossLM predictions, frozen `glosslm-v1` | [github.com/lecs-lab/polygloss](https://github.com/lecs-lab/polygloss/tree/frozen/glosslm-v1) |
| Evaluation | XCOMET-XL | [huggingface.co/Unbabel/XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) |
| Evaluation | XCOMET-XXL (exploratory) | [huggingface.co/Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) |
| Evaluation | BLEU and chrF++ | [sacreBLEU](https://github.com/mjpost/sacrebleu) |

### Dataset

**SIGMORPHON 2023 Shared Task on Interlinear Glossing:**
[github.com/sigmorphon/2023glossingST](https://github.com/sigmorphon/2023glossingST),
used at commit `259be10`. Gitksan training data is taken from `data_v1/` at the
pinned commit `190689a`. Check the dataset's licence before reusing its
sentences, which also appear in `results/`.

### Grammar sources

The grammar books are copyrighted and not included. The packages derived from
them, which are what the models actually saw, are in `materials/`. Bibliographic
details, and where to put the PDFs if you want to rebuild the packages, are in
[inputs/grammar_pdfs/README.md](inputs/grammar_pdfs/README.md).

### Core papers

- Ramos, R., Chimoto, E. A., ter Hoeve, M. and Schluter, N. (2025).
  **GRAMMAMT: Improving Machine Translation with Grammar-Informed In-Context Learning.** ACL 2025.
  [aclanthology.org/2025.acl-long.1447](https://aclanthology.org/2025.acl-long.1447/)
- Tanzer, G., Suzgun, M., Visser, E., Jurafsky, D. and Melas-Kyriazi, L. (2024).
  **A Benchmark for Learning to Translate a New Language from One Grammar Book.** ICLR 2024.
  [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/52d63f9e4b81f866bf69fb3c834aad47-Paper-Conference.pdf)
- Ginn, M., Moeller, S., Palmer, A., Stacey, A., Nicolai, G., Hulden, M. and Silfverberg, M. (2023).
  **Findings of the SIGMORPHON 2023 Shared Task on Interlinear Glossing.** SIGMORPHON 2023.
  [aclanthology.org/2023.sigmorphon-1.20](https://aclanthology.org/2023.sigmorphon-1.20/)
- Zhang, K., Choi, Y. M., Song, Z., He, T., Wang, W. Y. and Li, L. (2024).
  **Hire a Linguist!: Learning Endangered Languages in LLMs with In-Context Linguistic Descriptions.** Findings of ACL 2024.
  [aclanthology.org/2024.findings-acl.925](https://aclanthology.org/2024.findings-acl.925/)
- Zhou, Z., Levin, L., Mortensen, D. R. and Waibel, A. (2020).
  **Using Interlinear Glosses as Pivot in Low-Resource Multilingual Machine Translation.**
  [arXiv:1911.02709](https://arxiv.org/abs/1911.02709)

The full bibliography is in the thesis.

## How to Investigate and Run

There are three levels. Most readers only need the first.

### 1. Read the results (no setup)

Everything needed to inspect the study is already in the repository.

1. Read the [thesis](thesis/Cagatay_Iskenderoglu_Masters_Thesis.pdf), then
   [THESIS_GUIDE.md](THESIS_GUIDE.md).
2. Open [reports/matched_gold_v2/README.md](reports/matched_gold_v2/README.md)
   for the final tables, figures and audit.
3. To trace a single number, find its row in
   [condition_catalog.tsv](reports/matched_gold_v2/condition_catalog.tsv). It
   gives the config, the saved predictions in `results/` and the score files in
   `metrics/` for that condition. Each prediction record keeps the source
   sentence, the reference and the raw model output.

### 2. Verify and rebuild the reports (CPU only, no keys)

These commands recompute the tables and figures from the saved outputs and run
the consistency tests. They do not load models or call paid APIs. Use Linux or
macOS with Python 3.11; on Windows, use WSL, because some code uses `fcntl`.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/finalize_gold_v2.py
python scripts/render_gold_v2_appendix.py
python -m unittest discover -s tests -p "test_final_gold_v2.py" -v
python -m unittest discover -s tests -p "test_repository_layout.py" -v
python -m unittest discover -s tests -p "test_artifact_layout.py" -v
```

To run every test: `python -m unittest discover -s tests -v`.

### 3. Re-run generation and scoring (GPU and API key)

Only needed to reproduce the model outputs themselves. The configurations in
`configs/matched_gold_v2/` are frozen; run them as they are rather than
regenerating them.

**Dataset.** The code expects the dataset in a `Database/` folder next to the
repository folder:

```bash
mkdir -p ../Database
git clone https://github.com/sigmorphon/2023glossingST.git ../Database/2023glossingST
git -C ../Database/2023glossingST checkout 259be10
```

**Keys.** Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` (for
Gemini) and `HF_TOKEN` (for model downloads). Load it with
`set -a; source .env; set +a`.

**Generation.** Each job runs one group file from
`configs/matched_gold_v2/groups/`. There are 48 groups: one per model, language
and method, with the Qwen Tsez groups split into three batches. Outputs go to
`results/matched_gold_v2/`. The runner skips sentences that already have a saved
output, so to regenerate from scratch, work in a fresh copy or move the existing
result files aside first.

```bash
# Gemini 2.5 Flash Lite (CPU, OpenRouter)
python runners/matched_gold_v2/run.py --group configs/matched_gold_v2/groups/gemini25flashlite_gitksan_shot.json

# Qwen3-VL-8B-Instruct (one H100-class GPU, about 80 GB RAM)
python runners/matched_gold_v2/run.py --group configs/matched_gold_v2/groups/qwen3_gitksan_shot.json
```

Qwen3.5-9B needs a newer `transformers` than the version pinned here: 4.57.6
does not recognise the `qwen3_5` model type. Create a second environment with a
newer `transformers` for the `qwen35_*` groups;
[runners/qwen35/setup_qwen35_env.sh](runners/qwen35/setup_qwen35_env.sh) shows
how the original one was built. The Slurm scripts in `scripts/jobs/matched_gold_v2/`
are the exact jobs used on the LRZ cluster. Their paths are cluster-specific, but
they show the resources and command for each group.

**Scoring and analysis.**

```bash
python runners/matched_gold_v2/score.py --xcomet      # BLEU, chrF++ and XCOMET-XL
python runners/scoring/score_gold_v2_xcomet_xxl.py    # XCOMET-XXL (large GPU)
python runners/matched_gold_v2/analyze.py             # paired significance tests
python scripts/finalize_gold_v2.py                    # rebuild the reports
```

The full protocol, including decoding settings, image inputs and failure
handling, is in [docs/matched_gold_v2/README.md](docs/matched_gold_v2/README.md)
and in the reproducibility appendix of the thesis.

## Notes and Limitations

- Complete coverage does not mean every output is a valid translation, or that
  grammar context consistently helps. Read the limitations in
  [THESIS_GUIDE.md](THESIS_GUIDE.md) before reusing any numbers.
- The Lezgi sensitivity tables are descriptive. Separate post-hoc Qwen-only
  cluster-bootstrap tests are in the additional study; do not attach 87-row
  significance tests to filtered scores.
- XCOMET-XXL is a second evaluator of the same predictions, not another study,
  and is reported as exploratory.
- Historical outputs are kept for provenance only and must not be pooled with
  the primary study.

## Licence and Citation

The code is released under the [MIT Licence](LICENSE). The dataset, the model
outputs and the grammar-derived materials remain subject to the terms of their
original sources. To cite this work, use [CITATION.cff](CITATION.cff) or the
"Cite this repository" button on GitHub.
