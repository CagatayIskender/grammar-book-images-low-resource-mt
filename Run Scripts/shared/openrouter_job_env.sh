#!/bin/bash

OPENROUTER_KEY_FILE="${OPENROUTER_API_KEY_FILE:-/dss/dsshome1/07/ge92kun2/.config/grammamt/openrouter.env}"

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -r "${OPENROUTER_KEY_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${OPENROUTER_KEY_FILE}"
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "[ERROR] OPENROUTER_API_KEY is not set." >&2
  echo "[ERROR] Export it before sbatch or create ${OPENROUTER_KEY_FILE} with mode 600." >&2
  exit 1
fi

export OPENROUTER_API_KEY
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-none}"
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source venv/bin/activate
mkdir -p results/openrouter metrics/openrouter/gemini25flashlite slurm_outputs
