#!/bin/bash

set -eo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source "Run Scripts/cache_env.sh"
source venv/bin/activate

python - <<'PY'
from transformers import AutoConfig

model_id = "Qwen/Qwen3.5-9B"
try:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"OK: transformers recognizes {model_id} as {cfg.model_type}")
except Exception as exc:
    print("Current transformers does not recognize Qwen3.5 yet.")
    print("Upgrade transformers in a controlled maintenance step before running these jobs.")
    print()
    print("Suggested command:")
    print("  pip install --upgrade --pre transformers")
    print()
    print("Original error:")
    print(exc)
PY
