# Frontier-Evals for AiScientist

This repository documents the customized `frontier-evals` setup used inside
`AiScientist`, with a focus on running `PaperBench` from a vendored directory:

```text
AiScientist/
└── benchmark/
    └── frontier-evals/
```

This copy is not intended to be used as a standalone Git repository. It is
designed to live under `AiScientist/benchmark/frontier-evals`, and all Git and
Git LFS operations should be executed from the `AiScientist` repository root.

## What is included

This integration keeps a narrow, opinionated runtime surface:

- Solver families:
  - `AiScientist`
  - `BasicAgent`
  - `IterativeAgent`
- Supported solver backends:
  - `glm-5`
  - `gemini-3-flash-preview`
- Judge backend:
  - `gpt-5.4`

All preserved runner scripts write their outputs under:

```text
AiScientist/output_dir/run/paper_bench/
```

## Repository layout

The commands in this README assume the following structure:

```text
AiScientist/
├── benchmark/
│   └── frontier-evals/
│       └── project/
│           └── paperbench/
└── output_dir/
```

Define the following variables before running any commands:

```bash
export AISCIENTIST_ROOT="/path/to/AiScientist"
export FE_ROOT="${AISCIENTIST_ROOT}/benchmark/frontier-evals"
export PB_ROOT="${FE_ROOT}/project/paperbench"
```

## Prerequisites

You should have the following installed on the host machine:

- `git`
- `git-lfs`
- `uv`
- `python` support required by `uv sync --python=3.11`
- `docker` if you plan to prebuild runtime images or use container-based flows

Optional but commonly needed:

- A Hugging Face token for tasks that download gated assets
- Access to the model providers used by the solver and judge

## 1. Fetch Git LFS assets

Because `frontier-evals` is vendored into `AiScientist`, run all LFS commands
from the `AiScientist` repository root rather than from
`benchmark/frontier-evals`.

```bash
cd "${AISCIENTIST_ROOT}"

git lfs install
git lfs pull --include="benchmark/frontier-evals/project/paperbench/**"
git lfs checkout
```

This restores the large assets used by `PaperBench`, including materials under:

- `benchmark/frontier-evals/project/paperbench/data/papers/**`
- `benchmark/frontier-evals/project/paperbench/data/judge_eval/**`
- selected LFS-tracked experiment artifacts under `project/paperbench/experiments/**`

If you see files that still contain Git LFS pointer text such as:

```text
version https://git-lfs.github.com/spec/v1
```

then the LFS assets have not been fully restored yet.

## 2. Create the Python environment

Move into the `PaperBench` project directory and create the local environment:

```bash
cd "${PB_ROOT}"

uv sync --python=3.11
source .venv/bin/activate
```

For the current custom tooling stack used by this vendored setup, the following
extra packages are also recommended:

```bash
uv pip install protobuf==3.20.3
uv pip install omegaconf
uv pip install -U 'volcengine-python-sdk[ark]'
```

If your internal or private environment requires additional packages, install
them after the base `uv sync` step.

## 3. Set the base environment

At minimum, export the following variables:

```bash
export AISCIENTIST_ROOT="/path/to/AiScientist"
export FE_ROOT="${AISCIENTIST_ROOT}/benchmark/frontier-evals"
export PB_ROOT="${FE_ROOT}/project/paperbench"

export PAPERBENCH_DATA_DIR="${PB_ROOT}/data"
export HF_TOKEN="<optional_huggingface_token>"
export PB_TOOL_USER="<optional_tool_user>"
```

Notes:

- `PAPERBENCH_DATA_DIR` should point to `${PB_ROOT}/data`
- `HF_TOKEN` is optional, but some papers require it for model or dataset access
- `PB_TOOL_USER` is optional and only needed if your tooling stack uses it

## 4. Configure model credentials

### Option A: `glm-5` solver with `gpt-5.4` judge

```bash
export PB_GLM5_AZURE_OPENAI_ENDPOINT="<your_azure_compatible_endpoint>"
export PB_GLM5_AZURE_OPENAI_API_KEY="<your_glm5_key>"
export PB_GLM5_OPENAI_BASE_URL="<your_openai_compatible_base_url>"
export PB_JUDGE_OPENAI_API_KEY="<your_gpt54_judge_key>"
```

### Option B: `gemini-3-flash-preview` solver with `gpt-5.4` judge

```bash
export PB_GEMINI_API_KEY="<your_gemini_key>"
export PB_JUDGE_OPENAI_API_KEY="<your_gpt54_judge_key>"
```

The Gemini runner defaults to the Google OpenAI-compatible endpoint:

```bash
https://generativelanguage.googleapis.com/v1beta/openai/
```

If you need to override it:

```bash
export PB_GEMINI_OPENAI_BASE_URL="<your_gemini_openai_compatible_base_url>"
```

## 5. Common runtime knobs

These are optional, but they are the most useful runtime controls:

```bash
export PAPER_SPLIT="all"              # e.g. all, lite, debug, dev, subset1, subset2, compare
export RUN_TIME="86400"               # per-task wall clock limit, in seconds
export GPU_CANDIDATE_IDS="0,1,2,3"
export GPU_COUNT="1"
export GPU_AUTO_ALLOCATE="true"
export GPU_SHARE_MODE="true"
export RESUME_RUN_GROUP_ID=""
export MAX_RESPONSE_TOKENS="32768"
export JUDGE_MAX_TOKENS="16384"
```

For `AiScientist` runs, you can also choose the subagent profile:

```bash
export SUBAGENT_CONFIG_PROFILE="default"
```

## 6. Optional: prebuild Docker images

If you want to prepare the runtime images in advance:

```bash
cd "${PB_ROOT}"
bash paperbench/scripts/build-docker-images.sh
```

This builds:

- `pb-env`
- `pb-reproducer`

If your network requires a proxy, set `http_proxy` and `https_proxy` before
running the build script. The script forwards those values into `docker build`.

## 7. Run experiments

All commands below should be executed from the `PaperBench` root:

```bash
cd "${PB_ROOT}"
```

### AiScientist

GLM-5:

```bash
bash scripts/aiScientist/aisci_glm5.sh
```

Gemini:

```bash
bash scripts/aiScientist/aisci_gemini3.sh
```

### BasicAgent

GLM-5:

```bash
bash scripts/basicAgent/basic_all_run_glm5.sh
```

Gemini:

```bash
bash scripts/basicAgent/basic_all_run_gemini3_boe.sh
```

### IterativeAgent

GLM-5:

```bash
bash scripts/iterativeAgent/iterative_run_glm5.sh
```

Gemini:

```bash
bash scripts/iterativeAgent/iterative_run_gemini3_boe.sh
```

## 8. Output locations

All preserved runner scripts write to:

```text
${AISCIENTIST_ROOT}/output_dir/run/paper_bench/
```

The high-level structure is:

```text
output_dir/run/paper_bench/
└── <paper_split>_run/
    ├── aiscientist/
    ├── basicagent/
    └── iterativeagent/
```

Within each family, runs are grouped by model configuration and time limit. Log
files are typically written to:

```text
.../log/run_<timestamp>.log
```

The scripts also generate:

```text
${PB_ROOT}/paperbench/solvers/agent.env
```

This file is created automatically from the current shell environment and
normally does not need to be edited by hand.

## 9. Resume a run group

To resume an interrupted run group, set:

```bash
export RESUME_RUN_GROUP_ID="<existing_run_group_id>"
```

Then rerun the same script that created the original run.

## 10. Aggregate scores

This repository includes a score aggregation helper:

```text
${PB_ROOT}/scripts/eval/paper_bench.py
```

It:

- reads one or more `run-group` directories
- detects `re_grade/` automatically
- prefers regraded scores when available
- writes `all_result.json` into each run-group directory

### Aggregate a single run group

```bash
cd "${PB_ROOT}"

./.venv/bin/python scripts/eval/paper_bench.py \
  "${AISCIENTIST_ROOT}/output_dir/run/paper_bench/all_run/aiscientist/default_glm-5_gpt-5.4_86400/<run-group-id>"
```

### Aggregate the latest run group

Example for `AiScientist + GLM-5`:

```bash
cd "${PB_ROOT}"

RUN_BASE="${AISCIENTIST_ROOT}/output_dir/run/paper_bench/all_run/aiscientist/default_glm-5_gpt-5.4_86400"
LATEST_RUN_GROUP="$(ls -dt "${RUN_BASE}"/*run-group* | head -n 1)"

./.venv/bin/python scripts/eval/paper_bench.py "${LATEST_RUN_GROUP}"
```

The resulting summary file is written to:

```text
<run-group-dir>/all_result.json
```

## 11. Minimal end-to-end example

If you want the smallest working example for
`AiScientist + glm-5 + gpt-5.4`:

```bash
export AISCIENTIST_ROOT="/path/to/AiScientist"
export FE_ROOT="${AISCIENTIST_ROOT}/benchmark/frontier-evals"
export PB_ROOT="${FE_ROOT}/project/paperbench"

cd "${AISCIENTIST_ROOT}"
git lfs install
git lfs pull --include="benchmark/frontier-evals/project/paperbench/**"
git lfs checkout

cd "${PB_ROOT}"
uv sync --python=3.11
source .venv/bin/activate

export PAPERBENCH_DATA_DIR="${PB_ROOT}/data"
export PB_GLM5_AZURE_OPENAI_ENDPOINT="<your_endpoint>"
export PB_GLM5_AZURE_OPENAI_API_KEY="<your_glm5_key>"
export PB_GLM5_OPENAI_BASE_URL="<your_base_url>"
export PB_JUDGE_OPENAI_API_KEY="<your_gpt54_judge_key>"
export PAPER_SPLIT="all"

bash scripts/aiScientist/aisci_glm5.sh
```

## 12. Troubleshooting

- If `PAPERBENCH_DATA_DIR` is reported missing, verify that it points to
  `${PB_ROOT}/data`
- If a file still looks like a Git LFS pointer, rerun `git lfs pull` and
  `git lfs checkout` from the `AiScientist` root
- If custom web or tool-related modules fail to import, your environment is
  missing extra dependencies beyond the base `uv sync`
- If the judge fails, inspect the corresponding run-group logs and grader
  artifacts under the output directory
- If Docker builds are slow or failing, verify your network and proxy settings
