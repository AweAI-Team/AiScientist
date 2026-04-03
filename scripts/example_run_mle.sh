#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=/data03/chenguoxin.cgx/wksp/ai_workspace/dev/case/.env
OUTPUT_ROOT=/data03/chenguoxin.cgx/wksp/ai_workspace/output_dir/cli
COMPETITION_NAME=detecting-insults-in-social-commentary
COMPETITION_ZIP=/data03/chenguoxin.cgx/wksp/ai_workspace/dev/case/mle/detecting-insults-in-social-commentary.zip
LLM_PROFILE=glm-5
GPU_IDS=5
TIME_LIMIT="12h"
IMAGE=aisci-mle:test

uv run aisci \
  --env-file "${ENV_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  mle doctor
uv run aisci \
  --env-file "${ENV_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  mle run \
  --name "${COMPETITION_NAME}" \
  --zip "${COMPETITION_ZIP}" \
  --image "${IMAGE}" \
  --llm-profile "${LLM_PROFILE}" \
  --gpu-ids "${GPU_IDS}" \
  --time-limit "${TIME_LIMIT}" \
  --tui \
  --wait
