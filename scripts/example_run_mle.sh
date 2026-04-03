#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=/home/chenjie.survivi/aisci-0331/dev-AiSci/test/aisci_mle_glm5.env
OUTPUT_ROOT=/home/chenjie.survivi/aisci-0331/dev-AiSci/test/output
COMPETITION_NAME=detecting-insults-in-social-commentary
COMPETITION_ZIP=/home/chenjie.survivi/aisci-0331/dev-AiSci/data/detecting-insults-in-social-commentary.zip
LLM_PROFILE=glm-5
GPU_IDS=0
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
  --pull-policy never \
  --llm-profile "${LLM_PROFILE}" \
  --gpu-ids "${GPU_IDS}" \
  --time-limit "${TIME_LIMIT}" \
  --run-final-validation \
  --wait
