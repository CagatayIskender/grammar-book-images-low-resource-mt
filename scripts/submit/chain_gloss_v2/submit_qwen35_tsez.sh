#!/bin/bash
set -euo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
python3 scripts/submit_jobs.py --family chain_gloss_v2 --model qwen35 --group tsez --submit "$@"
