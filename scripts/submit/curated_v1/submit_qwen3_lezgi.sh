#!/bin/bash
set -euo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
python3 scripts/submit_jobs.py --family curated_v1 --model qwen3 --group lezgi --submit "$@"
