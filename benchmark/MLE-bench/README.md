# MLE-Bench — AiScientist evaluation harness

## Setup and run

How to install this repository, build Docker images, prepare data, run `run_agent.py` with the `aisci` agent image, and grade results. Agent IDs and defaults are in `agents/aisci/config.yaml` (for example `aisci/glm-5`, `aisci/gpt-5.2-responses`; `*-debug` profiles use shorter limits for quick checks). The Lite split is `experiments/splits/low.txt` (22 competitions; preparing data is large).

### 1. Host requirements

- **Docker** (Linux recommended) with enough disk for datasets and images.
- **NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)** if you use GPU-backed container configs.
- **Git** and **Git LFS** for competition assets.
- **Python** 3.10+ on the host for `mlebench` and `run_agent.py` (the agent runs inside the `aisci` container image).

### 2. Kaggle API

MLE-Bench downloads competition data via the Kaggle API.

1. Create API credentials from your Kaggle account settings.
2. Use either `~/.kaggle/kaggle.json` (mode `600`) **or** export `KAGGLE_USERNAME` and `KAGGLE_KEY` before `mlebench prepare` / `run_agent.py`.
3. In the browser, **accept the competition rules** on Kaggle for each competition you prepare (for example `spaceship-titanic` for a quick test).

### 3. Clone and fetch LFS

```bash
git clone <repository-url> mle-bench
cd mle-bench
git lfs fetch --all && git lfs pull
```

### 4. Python environment (host)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

mlebench prepare --help    # sanity check
```

Using **uv** is fine: `uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e .`.

### 5. Network, proxy, and Docker build (optional)

If the host needs an HTTP(S) proxy for apt/pip or image pulls:

- Export `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` as required.
- For builds, pass `--network=host` (so the build can reach `127.0.0.1` proxies) and proxy build-args, for example: `--build-arg http_proxy=$HTTP_PROXY --build-arg https_proxy=$HTTPS_PROXY --build-arg no_proxy=$NO_PROXY`

If **Docker Hub** is unreachable, obtain `ubuntu:20.04` (or your chosen base) by whatever mirror policy you use, then build `environment/Dockerfile` as usual.

### 6. Build the base image `mlebench-env`

From the **repository root**:

```bash
source .venv/bin/activate   # if you use a venv

docker build --platform=linux/amd64 \
  --network=host \
  --build-arg http_proxy=$HTTP_PROXY \
  --build-arg https_proxy=$HTTPS_PROXY \
  --build-arg no_proxy=$NO_PROXY \
  -t mlebench-env \
  -f environment/Dockerfile \
  .
```

The first build can take a long time.

### 7. Container runtime JSON (`--container-config`)

The default file `environment/config/container_configs/default.json` uses **`"runtime": "sysbox-runc"`**. Many machines **do not** have Sysbox; use a JSON that matches your Docker runtime (commonly **`nvidia`** or **`runc`**) and CPU/GPU limits.

Example for **one GPU** and **NVIDIA** runtime (adjust `nano_cpus`, `shm_size`, and `gpus` to your environment):

```json
{
    "mem_limit": null,
    "shm_size": "16G",
    "nano_cpus": 36000000000,
    "runtime": "nvidia",
    "gpus": 1
}
```

Save as e.g. `configs/container_config_gpu.json` and pass `--container-config /absolute/path/to/container_config_gpu.json` to `run_agent.py`. See `agents/run.py` for how these fields map to `docker run`.

### 8. Build the agent image `aisci`

Run from the **repo root**. The build **context** must be **`agents/aisci/`**. All `--build-arg` flags must appear **before** the final context path. After **any** change under `agents/aisci/`, rebuild this image (use `--no-cache` if you hit stale layers or missing `subagents/`).

```bash
cd /path/to/mle-bench
source .venv/bin/activate

docker build --platform=linux/amd64 \
  --network=host \
  --build-arg http_proxy=$HTTP_PROXY \
  --build-arg https_proxy=$HTTPS_PROXY \
  --build-arg no_proxy=$NO_PROXY \
  --build-arg SUBMISSION_DIR=/home/submission \
  --build-arg LOGS_DIR=/home/logs \
  --build-arg CODE_DIR=/home/code \
  --build-arg AGENT_DIR=/home/agent \
  -t aisci \
  -f agents/aisci/Dockerfile \
  agents/aisci/
```

Sanity check (override entrypoint so the grading server does not start):

```bash
docker run --rm --entrypoint "" aisci ls -la /home/agent/subagents/
```

### 9. Prepare competition data

**Smoke test (one small competition):**

```bash
mlebench prepare -c spaceship-titanic
ls ~/.cache/mle-bench/data/spaceship-titanic/prepared/
```

**Lite split (22 competitions, large download):**

```bash
mlebench prepare --lite
```

### 10. API credentials and environment variables

Export them in your shell (or a private `.env` that you `source`) before `run_agent.py`.

- **`agents/utils.py`** resolves `agents/aisci/config.yaml` placeholders like `${{ secrets.OPENAI_API_KEY }}` from the **current environment** (variable name matches the part after `secrets.`).
- Typical variables:
  - **Azure:** `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`
  - **OpenAI-compatible:** `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`
  - **Agent runtime:** `AISCI_API_MODE` (`completions` | `responses`), `AISCI_MODEL`, and optional `AISCI_WEB_SEARCH`, `AISCI_REASONING_EFFORT`, `AISCI_CONTEXT_REDUCE_STRATEGY`, etc. See `agents/aisci/llm_client.py` and `agents/aisci/config.yaml`.

| Mode | Env | Typical use |
|------|-----|-------------|
| Chat Completions | `AISCI_API_MODE=completions` | GLM, many OpenAI-compatible models |
| Responses | `AISCI_API_MODE=responses` | GPT with web search / reasoning where supported |

### 11. Run directory and ports

- Set **`MLEBENCH_RUNS_DIR`** if you want runs outside the default location.
- **`run_dir/logs/`** is mounted at `/home/logs` in the container (`agent.log`, `conversation.jsonl`, etc. update on the host during the run).
- If several `run_agent.py` processes share one host and **grading ports** conflict, set **`MLEBENCH_GRADING_PORT_BASE`** (e.g. `6000`) before starting.

### 12. Run jobs

**Smoke test** (single competition, debug profile):

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /path/to/mle-bench
source .venv/bin/activate

export MLEBENCH_RUNS_DIR="${MLEBENCH_RUNS_DIR:-/path/to/runs}"
# export KAGGLE_USERNAME=...  KAGGLE_KEY=...   if not using ~/.kaggle/kaggle.json

# --- LLM credentials (set according to your provider) ---
# export AZURE_OPENAI_ENDPOINT=...
# export AZURE_OPENAI_API_KEY=...
# export OPENAI_API_VERSION=...
# export AISCI_API_MODE=completions
# export AISCI_MODEL=glm-5
# export OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"   # if required by your stack

CONTAINER_CONFIG="/path/to/container_config_gpu.json"

python run_agent.py \
  --agent-id aisci/glm-5-debug \
  --competition-set experiments/splits/spaceship-titanic.txt \
  --container-config "${CONTAINER_CONFIG}" \
  --n-workers 1 \
  --n-seeds 1 \
  --gpu-ids 0
```

**Lite split (example: GLM-5, multiple GPUs):**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /path/to/mle-bench
source .venv/bin/activate

export MLEBENCH_RUNS_DIR="${MLEBENCH_RUNS_DIR:-/path/to/runs}"
# --- same Kaggle + LLM exports as above ---

CONTAINER_CONFIG="/path/to/container_config_gpu.json"

python run_agent.py \
  --agent-id aisci/glm-5 \
  --competition-set experiments/splits/low.txt \
  --container-config "${CONTAINER_CONFIG}" \
  --n-workers 8 \
  --n-seeds 1 \
  --gpu-ids 0,1,2,3,4,5,6,7
```

- **`--n-workers`**: concurrent tasks; may exceed GPU count if memory allows (least-loaded GPU assignment).
- For **GPT + Responses**, use an agent id such as `aisci/gpt-5.2-responses` and `AISCI_API_MODE=responses` if not already set via `config.yaml` / your exports.

### 13. Rebuild `aisci` after code changes

Edits under **`agents/aisci/`** require rebuilding the **`aisci`** image before new runs pick them up.

### 14. Grading

Use the run group directory printed when jobs finish (under `MLEBENCH_RUNS_DIR` or your default runs path). Example:

```bash
source .venv/bin/activate
RUNS_ROOT="${MLEBENCH_RUNS_DIR:-runs}"
RUN_GROUP="<timestamp>_run-group_aisci"

python experiments/make_submission.py \
  --metadata "${RUNS_ROOT}/${RUN_GROUP}/metadata.json" \
  --output "${RUNS_ROOT}/${RUN_GROUP}/submission.jsonl"

mlebench grade \
  --submission "${RUNS_ROOT}/${RUN_GROUP}/submission.jsonl" \
  --output-dir "${RUNS_ROOT}/${RUN_GROUP}"
```

For statistics over multiple seeds, run additional seeds and use `experiments/aggregate_grading_reports.py`.
