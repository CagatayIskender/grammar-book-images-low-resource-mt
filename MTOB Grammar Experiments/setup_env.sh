#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="${ROOT}/../venv/bin/python"
VENDOR="${ROOT}/vendor"

if [ ! -x "${PROJECT_PYTHON}" ]; then
  echo "[ERROR] Existing project Python is unavailable: ${PROJECT_PYTHON}" >&2
  exit 2
fi

mkdir -p "${VENDOR}"
"${PROJECT_PYTHON}" -m pip install \
  --target "${VENDOR}" \
  --no-deps \
  --upgrade \
  --requirement "${ROOT}/requirements.txt"

export HF_HOME="${ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${VENDOR}:${ROOT}:${PYTHONPATH:-}"
"${PROJECT_PYTHON}" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "sentence-transformers/all-mpnet-base-v2",
    allow_patterns=[
        "1_Pooling/*",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
)
PY

echo "[OK] Suite-local retrieval dependencies and embedding model are ready."
