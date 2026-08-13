#!/bin/bash
set -eo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT

LANGUAGE="${1:-}"

case "${LANGUAGE}" in
  gitksan)
    PATTERN="run_gitksan_qwen*.sh"
    ;;
  gitksan_pdf1)
    PATTERN="run_gitksan_qwen*pdf1_brown*.sh"
    ;;
  gitksan_pdf2)
    PATTERN="run_gitksan_qwen*pdf2_rigsby*.sh"
    ;;
  lezgi)
    PATTERN="run_lezgi_qwen*.sh"
    ;;
  natugu)
    PATTERN="run_natugu_qwen*.sh"
    ;;
  *)
    echo "[ERROR] Usage: bash \"Run Scripts/submit/submit_old_qwen3_language.sh\" {gitksan|gitksan_pdf1|gitksan_pdf2|lezgi|natugu}" >&2
    exit 2
    ;;
esac

mapfile -t JOBS < <(
  find "Run Scripts/qwen3" \
    -type f \
    -name "${PATTERN}" \
    ! -name "*newmaterials*" \
    ! -name "*score*" \
    ! -name "*_md.sh" \
    | sort
)

if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "[ERROR] No old Qwen 3 jobs found for ${LANGUAGE}." >&2
  exit 1
fi

echo "[INFO] Submitting ${#JOBS[@]} old Qwen 3 ${LANGUAGE} jobs"
mapfile -t QUEUED_NAMES < <(squeue -h -u "${USER}" -o "%j" || true)

submitted=0
skipped=0
for job in "${JOBS[@]}"; do
  job_name="$(basename "${job}")"
  already_queued=0
  for queued_name in "${QUEUED_NAMES[@]}"; do
    if [ "${queued_name}" = "${job_name}" ]; then
      already_queued=1
      break
    fi
  done
  if [ "${already_queued}" -eq 1 ]; then
    echo "[INFO] Skipping already queued job ${job_name}"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[INFO] Submitting ${job}"
  job_id="$(sbatch --parsable "${job}")"
  echo "[INFO] Submitted ${job_id}"
  submitted=$((submitted + 1))
done

echo "[INFO] old ${LANGUAGE} submit complete: submitted=${submitted}, skipped=${skipped}"
