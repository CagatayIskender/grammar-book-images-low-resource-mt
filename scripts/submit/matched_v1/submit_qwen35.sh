#!/bin/bash
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
venv/bin/python scripts/submit_matched_jobs.py --models qwen35 --submit "$@"
