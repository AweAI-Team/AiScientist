#!/usr/bin/env bash
set -euo pipefail

PAPER_MD=/data03/chenguoxin.cgx/wksp/ai_workspace/dev/case/paper/pinn/paper.md
LLM_PROFILE=glm-5
GPU_IDS=2
TIME_LIMIT="24h"

uv run aisci paper doctor
uv run aisci --env-file /data03/chenguoxin.cgx/wksp/ai_workspace/dev/case/.env \
  --output-root "/data03/chenguoxin.cgx/wksp/ai_workspace/output_dir/cli/" \
  paper run \
  --paper-md "${PAPER_MD}" \
  --image aisci_paper:test \
  --llm-profile "${LLM_PROFILE}" \
  --gpu-ids "${GPU_IDS}" \
  --time-limit "${TIME_LIMIT}" \
  --tui \
  --wait


# aisci --output-root /data03/chenguoxin.cgx/wksp/ai_workspace/output_dir/cli/ tui