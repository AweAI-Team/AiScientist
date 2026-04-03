# AiScientist

Independent AI Scientist workbench for `paper` and `mle` jobs.

## Quick Start

```bash
uv sync
uv run aisci --help
./.venv/bin/aisci --help
uv run aisci tui
```

## Runtime Config File

The CLI now auto-loads environment variables from the first matching files it
finds in the repo root or current directory:

- `.env`
- `.env.aisci`
- `.env.local`

Use `.env.example` as the template, then run the CLI directly without an extra wrapper script.

`scripts/run_paper_job.sh` remains available, but it is only a thin wrapper over `aisci paper run --wait`.

Recommended mental model:

- `AISCI_OUTPUT_ROOT`: where runtime outputs go, including `jobs/`, `export/`, and `.aisci/`
- `AISCI_REPO_ROOT`: where the AiScientist repo itself lives, used to find `config/` and `src/`

Most users only need `AISCI_OUTPUT_ROOT`. `AISCI_REPO_ROOT` is usually unnecessary unless you launch the installed CLI from outside the repo and the process cannot infer where the checked-out AiScientist source tree is.

LLM selection is driven by a YAML registry rather than hardcoded string parsing.
The default registry lives at `config/llm_profiles.yaml` in the repo root.
Users are expected to edit that top-level `config/` file directly rather than touching `src/...` internals.
If you want to maintain your own profile set, point `AISCI_LLM_PROFILE_FILE` at a custom YAML file.

Sandbox runtime image selection is also driven by a YAML registry.
The default registry lives at `config/image_profiles.yaml` in the repo root.
If you want to maintain your own image profile set, point `AISCI_IMAGE_PROFILE_FILE` at a custom YAML file.
The checked-in default `paper` profile assumes a locally prebuilt `aisci-paper:latest` image.
The local MLE build helper currently produces `aisci-mle:test`.

You can also point to a custom file explicitly:

```bash
./.venv/bin/aisci --env-file /abs/path/to/paper.env paper doctor
```

## Paper Mode Prerequisites

`paper` jobs will not start unless all of the following are true:

- Python `>=3.12`
- Docker daemon is reachable from the host
- the selected `--llm-profile` has all required backend env vars set in the host environment

Optional but recommended:

- `AISCI_OUTPUT_ROOT`: override where jobs, exports, and `.aisci/` state are written
- `AISCI_LLM_PROFILE_FILE`: optional override for the LLM profile registry YAML
- `AISCI_MAX_STEPS`: overrides the default paper loop step budget (`80`)
- `AISCI_REMINDER_FREQ`: overrides the default reminder interval (`5`)
- `AISCI_IMAGE_PROFILE_FILE`: optional override for the runtime image profile YAML
- `AISCI_REPO_ROOT`: advanced override for locating repo-local assets such as `config/` and `src/`

Sanity-check the local environment before running:

```bash
uv run aisci paper doctor
./.venv/bin/aisci paper doctor
```

## Paper Job Runtime Model

Current `paper` runs use a host-agent + Docker-sandbox split:

- The `aisci` worker process on the host runs the main agent loop and all subagents.
- A persistent Docker container is started as the code-execution sandbox.
- The host agent talks to that sandbox through shell/file operations.
- Final validation, if enabled, starts a fresh container from the same image and runs `bash reproduce.sh`.
- `src/aisci_domain_paper/orchestrator.py` is no longer the paper runtime entrypoint.

The sandbox still uses these canonical paths:

- `/home/paper`: staged paper inputs
- `/home/submission`: working repo and final `reproduce.sh`
- `/home/agent`: analysis and planning artifacts
- `/home/logs`: agent/runtime logs

This means:

- LLM keys stay on the host.
- `config/llm_profiles.yaml` is only read on the host.
- Docker only needs to execute commands and hold the isolated workspace.
- The run records both `workspace/agent/resolved_llm_config.json` and `state/sandbox_session.json` for debugging.

## Paper CLI

The main entrypoint is:

```bash
uv run aisci paper run [OPTIONS]
./.venv/bin/aisci paper run [OPTIONS]
```

At least one of the following inputs must be provided:

- `--pdf PATH`: path to a paper PDF
- `--paper-bundle-zip PATH`: zip extracted into `/home/paper`
- `--paper-md PATH`: markdown copy of the paper

Other `paper run` options exposed by the current CLI:

- `--llm-profile TEXT`: profile key from the YAML registry; if omitted, the registry's `defaults.paper` value is used
- `--gpus INT`: default `0`
- `--time-limit TEXT`: default `24h`; parsed from units like `30m`, `8h`, `1d12h`
- `--inputs-zip PATH`: extra context bundle extracted into `/home/paper`
- `--rubric-path PATH`: copied to `/home/paper/rubric.json`
- `--blacklist-path PATH`: copied to `/home/paper/blacklist.txt`
- `--addendum-path PATH`: copied to `/home/paper/addendum.md`
- `--submission-seed-repo-zip PATH`: extracted into `/home/submission`
- `--image TEXT`: Docker image ref for the sandbox runtime
- `--pull-policy TEXT`: one of `if-missing`, `always`, or `never`; if omitted, the selected image profile decides
- `--supporting-materials PATH`: repeatable option; each file is copied into `/home/paper`
- `--run-final-validation` / `--skip-final-validation`: default is enabled
- `--detach` / `--wait`: default is detached
- `--tui`: attach the live terminal dashboard; requires `--wait`

Current `paper run` CLI defaults that are not user-configurable from the CLI:

- `objective="paper reproduction job"`
- `enable_online_research=True`

If you need to change those values today, construct the `JobSpec` in Python.

## Recommended Paper Usage

Minimal run with a PDF:

```bash
cp .env.example .env
# edit .env and fill OPENAI_API_KEY or AZURE_OPENAI_API_KEY

./docker/build_paper_image.sh

./.venv/bin/aisci paper run \
  --pdf /abs/path/to/paper.pdf \
  --wait \
  --tui
```

A more complete run with staged context:

```bash
./.venv/bin/aisci paper run \
  --pdf /abs/path/to/paper.pdf \
  --inputs-zip /abs/path/to/context_bundle.zip \
  --rubric-path /abs/path/to/rubric.json \
  --addendum-path /abs/path/to/addendum.md \
  --submission-seed-repo-zip /abs/path/to/seed_repo.zip \
  --supporting-materials /abs/path/to/notes.md \
  --supporting-materials /abs/path/to/diagram.png \
  --time-limit 12h \
  --llm-profile gpt-5.4 \
  --pull-policy if-missing \
  --wait
```

If `--image` is omitted, `aisci` resolves the default runtime image from `config/image_profiles.yaml`.
The checked-in default points to `aisci-paper:latest`, which is what `docker/build_paper_image.sh` produces.
If you publish your sandbox image to a registry, either pass `--image <registry-ref>` directly or update `config/image_profiles.yaml`.

## LLM Profiles

The default registry schema is:

```yaml
defaults:
  paper: gpt-5.4
  mle: gpt-5.4

backends:
  openai:
    type: openai
    env:
      api_key:
        var: OPENAI_API_KEY
        required: true
      base_url:
        var: OPENAI_BASE_URL
  azure-openai:
    type: azure-openai
    env:
      endpoint:
        var: AZURE_OPENAI_ENDPOINT
        required: true
      api_key:
        var: AZURE_OPENAI_API_KEY
        required: true
      api_version:
        var: OPENAI_API_VERSION
        required: true

profiles:
  gpt-5.4:
    backend: openai
    model: gpt-5.4
    api: responses
    limits:
      max_completion_tokens: 131072
      context_window: 1000000
    features:
      use_phase: true

  glm-5:
    backend: azure-openai
    model: glm-5
    api: completions
    limits:
      max_completion_tokens: 65536
      context_window: 202752
    features:
      clear_thinking: true
```

Here `context_window` means the model's maximum context window. The runtime
derives its internal prune budget automatically; users do not need to configure
that separately.

Current provider backends supported by the code are:

- `openai`
- `azure-openai`

Current API modes supported by the code are:

- `responses`
- `completions`

For the default repo config, the only built-in model choices are:

- `gpt-5.4`
- `glm-5`

If you prefer background execution:

```bash
./.venv/bin/aisci paper run --pdf /abs/path/to/paper.pdf
```

Detached mode returns JSON containing the new `job_id`. Use that `job_id` for inspection commands below.

## Inspecting Runs

List jobs:

```bash
uv run aisci jobs list
```

Show one job with events and recorded artifacts:

```bash
uv run aisci jobs show <job_id>
```

List available logs for a run:

```bash
uv run aisci logs list <job_id>
```

Tail logs:

```bash
uv run aisci logs tail <job_id> --kind main
uv run aisci logs tail <job_id> --kind conversation
uv run aisci logs tail <job_id> --kind agent
uv run aisci logs tail <job_id> --kind subagent
uv run aisci logs tail <job_id> --kind validation
uv run aisci logs tail <job_id> --kind all
```

List recorded artifacts:

```bash
uv run aisci artifacts ls <job_id>
```

Export a job bundle:

```bash
uv run aisci export <job_id>
```

Open the terminal dashboard:

```bash
uv run aisci tui
uv run aisci tui job <job_id>
```

## Re-Run Validation Or Resume

Start a fresh self-check job from an existing paper run:

```bash
uv run aisci paper validate <job_id> --wait
```

Resume from an existing paper job spec:

```bash
uv run aisci paper resume <job_id> --wait
```

## Terminal UI

The terminal dashboard is the primary visualization surface:

```bash
uv run aisci tui
uv run aisci tui job <job_id>
uv run aisci paper run --pdf /abs/path/to/paper.pdf --wait --tui
```

Current TUI coverage:

- Jobs overview with live status, phase, latest event, and checks
- Single-job detail view for overview, events, logs, and results
- Dynamic GPU telemetry for jobs launched with `--gpu-ids`
- Small animated scientist mascot that tracks the selected job state

## Output Layout

Each run writes under `${AISCI_OUTPUT_ROOT:-<repo_root>}/jobs/<job_id>/`:

- `input/`: copied raw inputs
- `workspace/paper/`: staged paper materials
- `workspace/submission/`: working repo and final `reproduce.sh`
- `workspace/agent/`: summaries, plans, logs, self-check outputs
- `logs/`: job log, agent log, conversation log, subagent logs
- `artifacts/`: persisted `validation_report.json` and exported bundle metadata
- `export/`: zip bundle for the run

Useful files to inspect first for `paper` runs:

- `jobs/<job_id>/workspace/agent/paper_analysis/summary.md`
- `jobs/<job_id>/workspace/agent/prioritized_tasks.md`
- `jobs/<job_id>/workspace/submission/reproduce.sh`
- `jobs/<job_id>/workspace/agent/final_self_check.md`
- `jobs/<job_id>/logs/agent.log`

## What v1 Implements

- Unified SQLite-backed job store and filesystem layout under `jobs/<job_id>/`
- Shared Docker runtime API with per-mode default profiles
- Upstream-aligned `paper` AI Scientist loop: `read_paper -> prioritize_tasks -> implement -> run_experiment -> clean_reproduce_validation -> submit`
- `mle` job staging adapter with prompt-pack artifacts and validation plumbing
- CLI commands and a terminal dashboard for jobs, details, artifacts, and export

The paper mode now carries the default upstream AI Scientist execution path from
`paperbench`. Experimental or unhooked upstream modules are intentionally not
part of the alignment target. The `mle` mode remains focused on staging,
artifact generation, and runtime unification rather than a full upstream loop.

## Layout

- `src/aisci_core`: shared models, job store, worker, export
- `src/aisci_runtime_docker`: unified Docker runtime API
- `src/aisci_domain_paper`: paper-mode staging and validation
- `src/aisci_domain_mle`: mle-mode staging and validation
- `src/aisci_app`: CLI and terminal UI
