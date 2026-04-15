#!/bin/bash
set -x

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)"
conda activate agent

# ---- GPU detection ----
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')" 2>/dev/null || echo "WARNING: torch not available for GPU check"

echo "========================================="
echo "  AI Scientist Agent for MLE-Bench"
echo "========================================="
echo "Model:       ${AISCI_MODEL:-gpt-5.2-2025-12-11}"
echo "API mode:    ${AISCI_API_MODE:-completions}"
echo "Web search:  ${AISCI_WEB_SEARCH:-false}"
echo "Reasoning:   ${AISCI_REASONING_EFFORT:-none} (summary: ${AISCI_REASONING_SUMMARY:-none})"
echo "Time limit:  ${TIME_LIMIT_SECS}s"
echo "Max steps:   ${AISCI_MAX_STEPS:-500}"
echo "Config:      ${AISCI_CONFIG_PROFILE:-default}"
echo "Hardware:    ${HARDWARE}"
echo "Data dir:    /home/data/"
echo "Code dir:    /home/code/"
echo "Submit dir:  /home/submission/"
echo "========================================="

# ---- Ensure directories exist ----
mkdir -p /home/agent /home/code /home/submission

# ---- List competition data ----
echo "Competition data:"
ls -la /home/data/ 2>/dev/null || echo "  WARNING: /home/data/ not found"

# ---- Run orchestrator with timeout ----
# Add 120s buffer beyond the agent's internal time limit so _finalize() can
# write summary.json and copy log files before the shell kills the process.
SHELL_TIMEOUT=$((TIME_LIMIT_SECS + 120))
timeout --signal=TERM --kill-after=30 ${SHELL_TIMEOUT} python ${AGENT_DIR}/orchestrator.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
  echo "Timed out after ${SHELL_TIMEOUT}s (internal limit was ${TIME_LIMIT_SECS}s)"
fi

# ---- Post-run: ensure submission.csv exists ----
if [ ! -f /home/submission/submission.csv ]; then
  echo "WARNING: /home/submission/submission.csv not found after agent run"
  # Try to find it elsewhere
  for candidate in /home/code/submission.csv /home/code/output/submission.csv /home/code/submissions/submission.csv; do
    if [ -f "$candidate" ]; then
      echo "Found submission at $candidate, copying to /home/submission/submission.csv"
      cp "$candidate" /home/submission/submission.csv
      break
    fi
  done
fi

if [ -f /home/submission/submission.csv ]; then
  echo "Final submission:"
  wc -l /home/submission/submission.csv
  head -3 /home/submission/submission.csv
else
  echo "ERROR: No submission.csv found anywhere"
fi

exit $EXIT_CODE
