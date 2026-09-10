#!/bin/bash

export GRAMMAMT_DSS_ROOT="/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2"
export HF_HOME="${GRAMMAMT_DSS_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${GRAMMAMT_DSS_ROOT}/cache/torch"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}"

echo "[INFO] HF_HOME: ${HF_HOME}"
