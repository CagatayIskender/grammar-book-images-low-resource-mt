#!/bin/bash

set -eo pipefail

PROJECT_ROOT="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
BASE_VENV="${PROJECT_ROOT}/venv"
QWEN35_VENV="${QWEN35_VENV:-/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/envs/grammamt-qwen35}"

mkdir -p "$(dirname "${QWEN35_VENV}")"
python3 -m venv --clear --without-pip "${QWEN35_VENV}"

BASE_SITE="$("${BASE_VENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
QWEN35_SITE="$("${QWEN35_VENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"

printf '%s\n' "${BASE_SITE}" > "${QWEN35_SITE}/grammamt_base_venv.pth"

source "${QWEN35_VENV}/bin/activate"
"${BASE_VENV}/bin/python" -m pip install \
  --target "${QWEN35_SITE}" \
  --upgrade \
  --pre \
  --no-deps \
  transformers tokenizers==0.23.0rc0 safetensors qwen-vl-utils av \
  huggingface_hub hf-xet filelock fsspec tqdm \
  httpx==0.28.1 httpcore anyio h11 idna certifi sniffio exceptiongroup \
  typer shellingham rich annotated-doc pygments markdown-it-py mdurl

python - <<'PY'
from transformers import AutoConfig

model_id = "Qwen/Qwen3.5-9B"
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"OK: {model_id} recognized as model_type={cfg.model_type}")
PY
