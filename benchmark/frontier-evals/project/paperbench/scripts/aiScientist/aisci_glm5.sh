#!/bin/bash
set -euo pipefail

# Required:
#   PB_GLM5_AZURE_OPENAI_API_KEY
#   PB_GLM5_AZURE_OPENAI_ENDPOINT
#   PB_GLM5_OPENAI_BASE_URL
#   PB_JUDGE_OPENAI_API_KEY
# Optional:
#   PB_TOOL_USER

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

pb_enter_root
pb_setup_common_runtime
pb_setup_custom_web_env
pb_setup_glm5_solver_env
pb_setup_openai_judge_env
pb_write_agent_env

export GPU_CANDIDATE_IDS="${GPU_CANDIDATE_IDS:-0,1,2,3,4,5,6,7}"
export GPU_COUNT="${GPU_COUNT:-1}"
export GPU_AUTO_ALLOCATE="${GPU_AUTO_ALLOCATE:-true}"
export GPU_SHARE_MODE="${GPU_SHARE_MODE:-true}"
export SUBAGENT_CONFIG_PROFILE="${SUBAGENT_CONFIG_PROFILE:-default}"

timestamp="$(pb_timestamp)"

PAPER_SPLIT="${PAPER_SPLIT:-all}"
SOLVER_MODEL_NAME="${SOLVER_MODEL_NAME:-glm-5}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt-5.4}"
RUN_TIME="${RUN_TIME:-86400}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-32768}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-16384}"
RESUME_RUN_GROUP_ID="${RESUME_RUN_GROUP_ID:-}"
SAVE_PATH="${PAPERBENCH_OUTPUT_ROOT}/run/paper_bench/${PAPER_SPLIT}_run/aiscientist/${SUBAGENT_CONFIG_PROFILE}_${SOLVER_MODEL_NAME}_${JUDGE_MODEL_NAME}_${RUN_TIME}"
mkdir -p "${SAVE_PATH}/log" "${SAVE_PATH}/src"

cp "./paperbench/solvers/aiscientist/subagents/configs/${SUBAGENT_CONFIG_PROFILE}.py" \
    "${SAVE_PATH}/src/${SUBAGENT_CONFIG_PROFILE}_${timestamp}.py"

uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split="${PAPER_SPLIT}" \
    paperbench.runs_dir="${SAVE_PATH}" \
    ${RESUME_RUN_GROUP_ID:+paperbench.resume_run_group_id=${RESUME_RUN_GROUP_ID}} \
    paperbench.solver=paperbench.solvers.aiscientist:AiScientistSolver \
    paperbench.solver.completer_config=paperbench.solvers.basicagent.completer:AzureOpenAICompletionsTurnCompleterConfig \
    paperbench.solver.completer_config.model="${SOLVER_MODEL_NAME}" \
    paperbench.solver.completer_config.max_tokens="${MAX_RESPONSE_TOKENS}" \
    paperbench.solver.time_limit="${RUN_TIME}" \
    paperbench.solver.use_subagents=true \
    paperbench.solver.computer_runtime=paperbench.solvers.local_configs:LocalComputerRuntimeSingleGPU \
    paperbench.reproduction.computer_runtime=paperbench.solvers.local_configs:LocalComputerRuntimeSingleGPU \
    paperbench.reproduction.timeout="${RUN_TIME}" \
    runner.max_retries=0 \
    runner.concurrency=10 \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model="${JUDGE_MODEL_NAME}" \
    paperbench.judge.completer_config.max_tokens="${JUDGE_MAX_TOKENS}" \
    runner.recorder=nanoeval.json_recorder:json_recorder 2>&1 | tee "${SAVE_PATH}/log/run_${timestamp}.log"
