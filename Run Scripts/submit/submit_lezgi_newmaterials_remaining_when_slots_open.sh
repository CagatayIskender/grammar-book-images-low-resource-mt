#!/bin/bash
#SBATCH --job-name=lezgi_nm_submit
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_outputs/%j.out

set -uo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT

JOBS=(
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_modelgloss_newmaterials_summary_tables_jpg.sh"
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_modelgloss_newmaterials_summary_tables_txt.sh"
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_modelgloss_newmaterials_summary_text_txt.sh"
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_newmaterials_cheatsheet_jpg.sh"
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_newmaterials_cheatsheet_txt.sh"
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_newmaterials_summary_tables_jpg.sh"
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_newmaterials_summary_tables_txt.sh"
  "Run Scripts/qwen35/lezgi_cyrillic/run_lezgi_cyrillic_qwen35_newmaterials_summary_text_txt.sh"
)

remaining=("${JOBS[@]}")
while ((${#remaining[@]} > 0)); do
  retry=()
  for job in "${remaining[@]}"; do
    if output="$(sbatch --parsable "$job" 2>&1)"; then
      printf '[SUBMITTED] %s %s\n' "$output" "$job"
    elif [[ "$output" == *QOSMaxSubmitJobPerUserLimit* ]]; then
      printf '[WAITING] QOS submit limit: %s\n' "$job"
      retry+=("$job")
    else
      printf '[ERROR] %s: %s\n' "$job" "$output" >&2
      exit 1
    fi
  done
  remaining=("${retry[@]}")
  if ((${#remaining[@]} > 0)); then
    printf '[INFO] %d jobs remain; retrying in 300 seconds.\n' "${#remaining[@]}"
    sleep 300
  fi
done

echo '[INFO] All remaining Lezgi newmaterials jobs submitted.'
