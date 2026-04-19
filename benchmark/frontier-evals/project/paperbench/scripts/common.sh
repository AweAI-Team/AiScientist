#!/bin/bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAPERBENCH_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"
export FRONTIER_EVALS_ROOT="$(cd "${PAPERBENCH_ROOT}/../.." && pwd)"
export AISCIENTIST_ROOT="$(cd "${FRONTIER_EVALS_ROOT}/../.." && pwd)"
export PAPERBENCH_OUTPUT_ROOT="${AISCIENTIST_ROOT}/output_dir"
export PAPERBENCH_TMP_ROOT="${AISCIENTIST_ROOT}/tmp/paperbench"

pb_enter_root() {
    cd "${PAPERBENCH_ROOT}"
}

pb_timestamp() {
    date +%Y-%m-%dT%H-%M-%S-GMT
}

pb_require_env() {
    local missing=()
    local var
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            missing+=("${var}")
        fi
    done
    if ((${#missing[@]} > 0)); then
        printf 'Missing required environment variables: %s\n' "${missing[*]}" >&2
        return 1
    fi
}

pb_setup_common_runtime() {
    mkdir -p "${PAPERBENCH_OUTPUT_ROOT}" "${PAPERBENCH_TMP_ROOT}"
    export PAPERBENCH_GRADE_TMPDIR="${PAPERBENCH_GRADE_TMPDIR:-${PAPERBENCH_TMP_ROOT}/grade_tmp}"
    mkdir -p "${PAPERBENCH_GRADE_TMPDIR}"
}

pb_setup_custom_web_env() {
    export LINK_SUMMARY_MODEL="${PB_LINK_SUMMARY_MODEL:-glm_5}"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
    if [[ -n "${PB_TOOL_USER:-}" ]]; then
        export BANDAI_USER="${PB_TOOL_USER}"
    fi
}

pb_setup_glm5_solver_env() {
    local glm_endpoint glm_key glm_openai_key
    glm_endpoint="${PB_GLM5_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT:-}}"
    glm_key="${PB_GLM5_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY:-}}"
    glm_openai_key="${PB_GLM5_OPENAI_API_KEY:-${glm_key}}"

    export AZURE_OPENAI_ENDPOINT="${glm_endpoint}"
    export AZURE_OPENAI_API_KEY="${glm_key}"
    export OPENAI_API_VERSION="${PB_GLM5_OPENAI_API_VERSION:-${OPENAI_API_VERSION:-2024-02-01}}"
    export OPENAI_BASE_URL="${PB_GLM5_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}"
    export OPENAI_API_KEY="${glm_openai_key}"

    pb_require_env AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_KEY
}

pb_setup_gemini_solver_env() {
    export OPENAI_BASE_URL="${PB_GEMINI_OPENAI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
    export OPENAI_API_KEY="${PB_GEMINI_API_KEY:-${GEMINI_API_KEY:-}}"

    pb_require_env OPENAI_API_KEY
}

pb_setup_openai_judge_env() {
    export GRADER_OPENAI_API_KEY="${PB_JUDGE_OPENAI_API_KEY:-${GRADER_OPENAI_API_KEY:-}}"
    pb_require_env GRADER_OPENAI_API_KEY
}

pb_write_agent_env() {
    cat > "${PAPERBENCH_ROOT}/paperbench/solvers/agent.env" <<EOF
OPENAI_BASE_URL=${OPENAI_BASE_URL:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-}
AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:-}
OPENAI_API_VERSION=${OPENAI_API_VERSION:-}
HF_TOKEN=${HF_TOKEN:-}
HTTP_PROXY=${HTTP_PROXY:-}
HTTPS_PROXY=${HTTPS_PROXY:-}
http_proxy=${http_proxy:-}
https_proxy=${https_proxy:-}
NO_PROXY=${NO_PROXY:-}
no_proxy=${no_proxy:-}
LINK_SUMMARY_MODEL=${LINK_SUMMARY_MODEL:-}
BANDAI_USER=${BANDAI_USER:-}
EOF
}
