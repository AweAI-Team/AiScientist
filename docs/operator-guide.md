# AiScientist Operator Guide

This guide keeps the operator-facing details out of the main README. Use it for host setup, GPU and Docker prerequisites, runtime caveats, build steps, doctor checks, run commands, lifecycle helpers, and configuration references.

## Prerequisites

Host-side requirements:

- Python 3.12+
- Docker with a reachable daemon
- `uv`
- API credentials for at least one configured LLM backend
- Optional NVIDIA GPUs if you want GPU-bound runs

Repository setup:

```bash
git clone https://github.com/AweAI-Team/AiScientist.git
cd AiScientist
uv sync --dev
```

## GPU-Enabled Docker

If you want to run AiScientist with `--gpu-ids` or `--gpus`, Docker must be able to pass through NVIDIA GPUs on the host.

Typical Linux/NVIDIA setup:

- Install a working NVIDIA GPU driver on the host.
- Install the NVIDIA Container Toolkit.
- Configure Docker with `sudo nvidia-ctk runtime configure --runtime=docker`.
- Restart Docker.
- Verify GPU passthrough with a simple `docker run --rm --gpus all ... nvidia-smi` check before launching AiScientist.

See the official NVIDIA guide for the latest installation steps:
[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

If you do not need GPU-bound runs, you can skip this and run CPU-only jobs.

## Environment and Credentials

Start from the shipped template:

```bash
cp .env.example .env
```

Fill credentials for at least one backend in [`.env.example`](../.env.example). The current setup assumes either OpenAI or Azure OpenAI style credentials.

Useful environment and profile notes:

- The shipped LLM defaults are not symmetric: `paper=glm-5`, `mle=gpt-5.4` in [`config/llm_profiles.yaml`](../config/llm_profiles.yaml).
- If you only have `OPENAI_API_KEY`, run paper commands with `--llm-profile gpt-5.4`.
- For paper doctor on OpenAI-only setups, set `AISCI_PAPER_DOCTOR_PROFILE=gpt-5.4`.
- Job state lives under the repo root by default. Use `AISCI_OUTPUT_ROOT` or `--output-root` if you want `jobs/` and `.aisci/` elsewhere.

Useful host-side knobs:

- `--env-file /path/to/.env`
- `--output-root /abs/path/to/runtime_root`
- `--llm-profile-file /abs/path/to/llm_profiles.yaml`
- `--image-profile-file /abs/path/to/image_profiles.yaml`
- `--gpu-ids 0,1` or `--gpus 2`

## Image Build

> [!IMPORTANT]
> The current Dockerfiles are tuned for our operator environment. Both [`docker/paper-agent.Dockerfile`](../docker/paper-agent.Dockerfile) and [`docker/mle-agent.Dockerfile`](../docker/mle-agent.Dockerfile) reference internal Ubuntu images and package mirrors. If you are outside that environment, replace those base-image and mirror lines before the first build.

Build the intended default runtime images:

```bash
bash docker/build_paper_image.sh
bash docker/build_mle_image.sh
```

Expected local image tags:

- `aisci-paper:latest`
- `aisci-mle:test`

These tags are used directly in the public quick starts below to avoid relying on profile defaults that you may want to change locally.

## Doctor Checks

Run the built-in health checks before your first live run:

```bash
AISCI_PAPER_DOCTOR_PROFILE=gpt-5.4 uv run aisci paper doctor
uv run aisci mle doctor
```

If you use the shipped Azure-backed `glm-5` paper profile, you can drop the `AISCI_PAPER_DOCTOR_PROFILE` override.

## Paper Run

Canonical Markdown-first paper reproduction flow:

```bash
uv run aisci --env-file .env paper run \
  --paper-md /abs/path/to/paper.md \
  --image aisci-paper:latest \
  --llm-profile gpt-5.4 \
  --gpu-ids 0 \
  --time-limit 24h \
  --wait \
  --tui
```

Other paper entrypoints:

- `--zip /abs/path/to/paper_bundle_or_context.zip`
- `--submission-seed-repo-zip /abs/path/to/starter_repo.zip`
- `--supporting-materials /abs/path/to/extra_note.md`

Paper zip note:

- `paper run` accepts one primary `--zip`; if you previously staged multiple archives into `/home/paper`, combine them locally before launching the job.

Expected paper outputs:

- `workspace/agent/paper_analysis/summary.md`
- `workspace/agent/paper_analysis/structure.md`
- `workspace/agent/prioritized_tasks.md`
- `workspace/agent/impl_log.md`
- `workspace/agent/exp_log.md`
- `workspace/agent/final_self_check.md`
- `artifacts/validation_report.json`
- `export/<job_id>.zip`

## MLE Run

Canonical self-contained MLE flow with a local competition zip:

```bash
uv run aisci --env-file .env mle run \
  --zip /abs/path/to/detecting-insults-in-social-commentary.zip \
  --name detecting-insults-in-social-commentary \
  --image aisci-mle:test \
  --llm-profile gpt-5.4 \
  --gpu-ids 0 \
  --time-limit 12h \
  --wait \
  --tui
```

Input selection notes:

- Prefer `--zip` for an offline, self-contained entrypoint.
- Use `--name` alone when you already have a prepared MLE-Bench cache.
- Use `--data-dir`, `--workspace-zip`, or `--competition-bundle-zip` for operator or migration flows.
- If the zip stem and competition slug differ, keep `--name` so the run keeps the correct registry id.

Expected MLE outputs:

- `workspace/agent/analysis/summary.md`
- `workspace/agent/prioritized_tasks.md`
- `workspace/agent/impl_log.md`
- `workspace/agent/exp_log.md`
- `workspace/submission/submission.csv`
- `workspace/submission/submission_registry.jsonl`
- `artifacts/validation_report.json` when final validation is enabled
- `export/<job_id>.zip`

## Inspect / Export / Resume / Validate

Core job inspection commands:

```bash
uv run aisci jobs list
uv run aisci jobs show <job_id>
uv run aisci logs list <job_id>
uv run aisci logs tail <job_id> --kind conversation
uv run aisci artifacts ls <job_id>
uv run aisci export <job_id>
```

Terminal-native monitoring:

```bash
uv run aisci tui
uv run aisci tui job <job_id>
```

Lifecycle helpers:

```bash
uv run aisci paper validate <job_id> --wait
uv run aisci paper resume <job_id> --wait
uv run aisci mle validate <job_id> --wait
uv run aisci mle resume <job_id> --wait
```

## Config Registry Files

- [`.env.example`](../.env.example): backend credentials, optional proxy variables, optional Hugging Face token, output-root overrides, and doctor flags.
- [`config/llm_profiles.yaml`](../config/llm_profiles.yaml): shared model registry and per-domain defaults.
- [`config/image_profiles.yaml`](../config/image_profiles.yaml): runtime image registry and pull policy defaults.
- [`config/paper_subagents.yaml`](../config/paper_subagents.yaml): paper-mode subagent step budgets and bash timeouts.

## Example Scripts

Operator examples already live in the repo:

- [`scripts/example_run_paper.sh`](../scripts/example_run_paper.sh)
- [`scripts/example_run_mle.sh`](../scripts/example_run_mle.sh)

The paper example script is the Markdown-based path: it launches `paper run --paper-md ...`.

## Caveats

- The control plane is generic Python plus Docker orchestration, but the bundled image recipes still carry internal infrastructure assumptions.
- The default paper image path is local-first after you build `aisci-paper:latest`.
- The current shared MLE image profile is remote-first, so public quick starts should pass `--image aisci-mle:test` explicitly or update [`config/image_profiles.yaml`](../config/image_profiles.yaml).
- Job state lives under the repo root by default. Use `AISCI_OUTPUT_ROOT` or `--output-root` if you want `jobs/` and `.aisci/` somewhere else.

## Use With Coding Agents

If you use Codex, Claude Code, or another coding agent, you can hand it this prompt:

```text
Help me clone AiScientist, adapt the Dockerfiles if this machine cannot use the current base images, fill .env from .env.example, build the paper and mle images, run doctor for both modes, and then launch either a paper or mle job.
```
