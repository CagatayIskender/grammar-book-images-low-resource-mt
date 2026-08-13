Qwen3.5 Experiments
===================

This folder is intentionally separate from the existing GRAMMAMT run scripts.
Nothing in the existing Qwen/Qwen3-VL setup is changed by these files.

Target model:

Qwen/Qwen3.5-9B

Important environment note
--------------------------

The current project virtual environment has transformers 4.57.6. It supports
Qwen3-VL, but it does not currently recognize the Qwen3.5 model type:

model_type: qwen3_5

So these scripts are prepared for Qwen3.5, but they should be run from a
separate Qwen3.5 environment rather than the working Qwen3-VL environment.

Create that environment once with:

bash "Qwen3.5 Experiments/setup_qwen35_env.sh"

By default it creates:

/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/envs/grammamt-qwen35

The Qwen3.5 environment overlays the existing GRAMMAMT venv for shared
packages, but installs newer Qwen3.5-compatible transformer packages into the
separate environment.

Run scripts
-----------

From the GRAMMAMT project root:

sbatch "Qwen3.5 Experiments/Run Scripts/run_natugu_qwen35_summary_text.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_natugu_qwen35_summary_modelinput_all.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_natugu_qwen35_pdfpages_modelinput_all.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_natugu_qwen35_summary_combined_image.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_natugu_qwen35_modelgloss_summary_modelinput_all.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_natugu_qwen35_modelgloss_pdfpages_modelinput_all.sh"

The scripts write Qwen3.5-specific output names under the normal metrics/ and
results/ directories.

Gitksan Qwen3.5 scripts
-----------------------

sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf1_brown_pdfpages_all_jpg.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf1_brown_summary_tables.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf1_brown_summary_text.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf1_brown_cheatsheet.sh"

sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf2_rigsby_pdfpages_all_jpg.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf2_rigsby_summary_tables.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf2_rigsby_summary_text.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_pdf2_rigsby_cheatsheet.sh"

sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf1_brown_pdfpages_all_jpg.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf1_brown_summary_tables.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf1_brown_summary_text.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf1_brown_cheatsheet.sh"

sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf2_rigsby_pdfpages_all_jpg.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf2_rigsby_summary_tables.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf2_rigsby_summary_text.sh"
sbatch "Qwen3.5 Experiments/Run Scripts/run_gitksan_qwen35_modelgloss_pdf2_rigsby_cheatsheet.sh"
