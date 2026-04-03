from __future__ import annotations

import re
from pathlib import Path

from aisci_core.models import RuntimeProfile, WorkspaceLayout
import yaml

from aisci_domain_mle.contracts import (
    DockerBuildPolicy,
    DockerMountPlan,
    InputSourceKind,
    LegacyPreparePlan,
    MLEPhase1JobSpec,
    ResolvedInputState,
    RuntimeOrchestrationPlan,
    RuntimeWorkspaceLayout,
    ValidationKind,
    ValidationPlan,
)
from aisci_domain_mle.mlebench_compat import (
    find_legacy_mlebench_repo_root,
    resolve_legacy_mlebench_repo_root,
)
from aisci_domain_mle.shared_infra_bridge import (
    build_mle_session_env,
    build_mle_sandbox_env,
    present_proxy_env_keys,
    preview_shared_mle_runtime,
)
from aisci_domain_mle.vendored_lite import vendored_lite_repo_root

AISCI_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCKER_BINARY = "docker"
DEFAULT_FORMAT_ONLY_VALIDATION = (
    "python - <<'PY'\n"
    "import csv\n"
    "with open('/home/submission/submission.csv', newline='') as f:\n"
    "    rows = list(csv.reader(f))\n"
    "print(f'rows={len(rows)}')\n"
    "PY"
)
SENSITIVE_ENV_KEYS = {
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}


def normalize_build_policy(raw: str | DockerBuildPolicy | None) -> DockerBuildPolicy:
    if isinstance(raw, DockerBuildPolicy):
        return raw
    value = (raw or DockerBuildPolicy.AUTO.value).strip().lower()
    try:
        return DockerBuildPolicy(value)
    except ValueError as exc:
        raise ValueError(f"invalid build policy {raw!r}; expected auto, force, or never") from exc


def parse_duration_seconds(text: str) -> int:
    value = (text or "24h").strip().lower()
    total = 0
    for amount, unit in re.findall(r"(\d+)([smhd])", value):
        n = int(amount)
        total += {
            "s": n,
            "m": n * 60,
            "h": n * 3600,
            "d": n * 86400,
        }[unit]
    if total <= 0:
        raise ValueError(f"invalid time limit {text!r}; expected values such as 45m, 12h, or 1d")
    return total


def sanitize_slug(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9][^a-z0-9-]*", "-", (value or "").strip().lower())
    cleaned = re.sub(r"[^a-z0-9-]+", "-", cleaned).strip("-")
    return cleaned or fallback


def build_runtime_workspace(run_root: str | Path) -> RuntimeWorkspaceLayout:
    root = Path(run_root).expanduser().resolve()
    workspace = root / "workspace"
    return RuntimeWorkspaceLayout(
        run_root=str(root),
        input_dir=str(root / "input"),
        workspace_dir=str(workspace),
        data_dir=str(workspace / "data"),
        code_dir=str(workspace / "code"),
        submission_dir=str(workspace / "submission"),
        agent_dir=str(workspace / "agent"),
        workspace_logs_dir=str(workspace / "logs"),
        host_logs_dir=str(root / "logs"),
    )


def _hardware_label(gpu_ids: list[str]) -> str:
    return f"gpu:{','.join(gpu_ids)}" if gpu_ids else "cpu"


def _mount_from_session(source: Path, target: str, *, read_only: bool) -> DockerMountPlan:
    return DockerMountPlan(
        source=str(source.resolve()),
        target=target,
        mode="ro" if read_only else "rw",
    )


def _competition_metadata_paths(
    competition_name: str,
    cache_root: Path,
) -> tuple[Path | None, Path | None]:
    legacy_repo = find_legacy_mlebench_repo_root()
    if legacy_repo is None:
        return None, None

    config_path = (
        legacy_repo / "mlebench" / "competitions" / competition_name / "config.yaml"
    ).resolve()
    if not config_path.is_file():
        return None, None

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    description_raw = str(payload.get("description") or "").strip()
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    sample_raw = str(dataset.get("sample_submission") or "").strip()

    description_path = (
        (legacy_repo / description_raw).resolve()
        if description_raw
        else None
    )
    sample_submission_path = (
        (cache_root / sample_raw).resolve()
        if sample_raw
        else None
    )
    return (
        description_path if description_path and description_path.is_file() else None,
        sample_submission_path if sample_submission_path and sample_submission_path.is_file() else None,
    )


def _optional_existing_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    return path if path.is_file() else None


def _safe_public_data_dir(source: Path) -> Path | None:
    source = source.expanduser().resolve()
    if source.name == "public" and source.is_dir():
        return source
    if source.name == "prepared" and (source / "public").is_dir():
        return (source / "public").resolve()
    if (source / "prepared" / "public").is_dir():
        return (source / "prepared" / "public").resolve()
    return None


def _host_setup_commands(
    workspace: RuntimeWorkspaceLayout,
    *,
    job_spec: MLEPhase1JobSpec,
    resolved_inputs: ResolvedInputState,
    description_path: Path | None = None,
    sample_submission_path: Path | None = None,
) -> list[list[str]]:
    run_root = Path(workspace.run_root).resolve()
    commands = [
        [
            "mkdir",
            "-p",
            str(run_root),
            workspace.input_dir,
            workspace.workspace_dir,
            workspace.data_dir,
            workspace.code_dir,
            workspace.submission_dir,
            workspace.agent_dir,
            workspace.workspace_logs_dir,
            workspace.host_logs_dir,
            str(run_root / "artifacts"),
            str(run_root / "export"),
            str(run_root / "state"),
        ]
    ]
    if job_spec.mode_spec.data_dir:
        public_dir = _safe_public_data_dir(Path(job_spec.mode_spec.data_dir))
        if public_dir is None:
            return commands
        commands.extend(
            [
                ["rm", "-rf", workspace.data_dir],
                ["mkdir", "-p", workspace.data_dir],
                ["cp", "-a", f"{public_dir}/.", workspace.data_dir],
            ]
        )
        if description_path is not None:
            commands.append(["cp", "-f", str(description_path), f"{workspace.data_dir}/description.md"])
        if sample_submission_path is not None:
            commands.append(
                [
                    "cp",
                    "-f",
                    str(sample_submission_path),
                    f"{workspace.data_dir}/sample_submission.csv",
                ]
            )
    elif (
        resolved_inputs.source_kind == InputSourceKind.COMPETITION_NAME
        and resolved_inputs.cache_prepared_exists
        and resolved_inputs.cache_prepared_dir
    ):
        public_dir = Path(resolved_inputs.cache_prepared_dir).resolve() / "public"
        commands.extend(
            [
                ["rm", "-rf", workspace.data_dir],
                ["mkdir", "-p", workspace.data_dir],
                ["cp", "-a", f"{public_dir}/.", workspace.data_dir],
            ]
        )
        if description_path is not None:
            commands.append(["cp", "-f", str(description_path), f"{workspace.data_dir}/description.md"])
        if sample_submission_path is not None:
            commands.append(
                [
                    "cp",
                    "-f",
                    str(sample_submission_path),
                    f"{workspace.data_dir}/sample_submission.csv",
                ]
            )
    return commands


def _shared_runtime_profile(job_spec: MLEPhase1JobSpec) -> RuntimeProfile:
    return RuntimeProfile(
        gpu_count=job_spec.runtime_profile.gpu_count,
        gpu_ids=list(job_spec.runtime_profile.gpu_ids),
        time_limit=job_spec.runtime_profile.time_limit,
        run_final_validation=job_spec.runtime_profile.run_final_validation,
        network_policy=job_spec.runtime_profile.network_policy,
        workspace_layout=WorkspaceLayout.MLE,
    )


def _preview_job_id(job_spec: MLEPhase1JobSpec, workspace: RuntimeWorkspaceLayout) -> str:
    candidate = job_spec.mode_spec.competition_name or Path(workspace.run_root).name or "mle-preview"
    return sanitize_slug(candidate, fallback="mle-preview")


def _command_container_name(command: list[str]) -> str:
    try:
        index = command.index("--name")
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return command[index + 1]


def _legacy_grade_plan(
    *,
    competition_name: str,
    cache_root: Path,
    workspace: RuntimeWorkspaceLayout,
    cleanup_command: list[str] | None = None,
) -> ValidationPlan:
    legacy_repo = resolve_legacy_mlebench_repo_root()
    if legacy_repo == vendored_lite_repo_root():
        grade_command = [
            "python3",
            "-m",
            "aisci_domain_mle.vendored_lite_cli",
            "grade-sample",
            str(Path(workspace.submission_dir) / "submission.csv"),
            competition_name,
            "--data-dir",
            str(cache_root),
        ]
        validation_description = (
            "Built-in vendored MLE-bench Lite grade-sample command against prepared cache data."
        )
        env: dict[str, str] = {}
        cwd = str(AISCI_REPO_ROOT)
    else:
        grade_command = [
            "python3",
            "-m",
            "mlebench.cli",
            "grade-sample",
            str(Path(workspace.submission_dir) / "submission.csv"),
            competition_name,
            "--data-dir",
            str(cache_root),
        ]
        validation_description = "Legacy MLE-Bench grade-sample command against prepared cache data."
        env = {"PYTHONPATH": str(legacy_repo)}
        cwd = str(legacy_repo)
    return ValidationPlan(
        kind=ValidationKind.LEGACY_GRADE,
        description=validation_description,
        passthrough_env_keys=[],
        cwd=cwd,
        env=env,
        host_command=grade_command,
        docker_cleanup_command=list(cleanup_command) if cleanup_command else None,
    )


def _shared_validation_plan(preview) -> ValidationPlan:
    return ValidationPlan(
        kind=ValidationKind.FORMAT_ONLY,
        description="Format-only validation preview through shared DockerRuntimeManager.run_validation().",
        passthrough_env_keys=[],
        docker_run_command=list(preview.validation_run_command or []),
        docker_exec_command=list(preview.validation_exec_command or []),
        docker_cleanup_command=list(preview.validation_cleanup_command or []),
    )


def _redact_value(key: str, value: str) -> str:
    if key in SENSITIVE_ENV_KEYS:
        return "<redacted>"
    return value


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    return {key: _redact_value(key, value) for key, value in env.items()}


def _redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    for item in command:
        if "=" not in item:
            redacted.append(item)
            continue
        key, value = item.split("=", 1)
        redacted.append(f"{key}={_redact_value(key, value)}")
    return redacted


def build_runtime_plan(
    *,
    job_spec: MLEPhase1JobSpec,
    resolved_inputs: ResolvedInputState,
    run_root: str | Path,
    build_policy: str | DockerBuildPolicy = DockerBuildPolicy.AUTO,
    dockerfile_path: str | None = None,
    docker_binary: str = DEFAULT_DOCKER_BINARY,
    image_tag: str | None = None,
    container_name: str | None = None,
) -> RuntimeOrchestrationPlan:
    policy = normalize_build_policy(build_policy)
    workspace = build_runtime_workspace(run_root)
    warnings: list[str] = [
        "runtime-plan is a migration helper preview. Authoritative live execution behavior comes from `aisci mle run` and the shared adapter.",
        "Current live architecture runs the MLE orchestrator on the host and uses Docker only for the /home sandbox workspace."
    ]
    prepare_plan: LegacyPreparePlan | None = None
    ready_to_execute = True
    description_path: Path | None = None
    sample_submission_path: Path | None = None

    if policy != DockerBuildPolicy.AUTO:
        warnings.append(
            "runtime-plan no longer owns local Docker builds; --build-policy is kept for compatibility only."
        )
    if dockerfile_path:
        warnings.append(
            "runtime-plan no longer owns dockerfile/image builds; --dockerfile is ignored in favor of shared image profiles."
        )
    if image_tag:
        warnings.append(
            "runtime-plan no longer overrides the image tag locally; --image-tag is ignored in favor of the shared image profile."
        )
    if container_name:
        warnings.append(
            "runtime-plan no longer precomputes container names; --container-name is ignored because the shared runtime owns session naming."
        )

    if resolved_inputs.source_kind == InputSourceKind.COMPETITION_NAME and not resolved_inputs.cache_prepared_exists:
        ready_to_execute = False
        prepare_plan = resolved_inputs.legacy_prepare_plan
        warnings.append(
            "prepared cache is missing; run the legacy prepare plan before the shared container session can start."
        )
    elif (
        resolved_inputs.source_kind == InputSourceKind.COMPETITION_NAME
        and resolved_inputs.cache_prepared_exists
        and resolved_inputs.competition_name
    ):
        description_override = _optional_existing_path(job_spec.mode_spec.description_path)
        sample_submission_override = _optional_existing_path(job_spec.mode_spec.sample_submission_path)
        metadata_description_path, metadata_sample_submission_path = _competition_metadata_paths(
            resolved_inputs.competition_name,
            Path(resolved_inputs.cache_root),
        )
        description_path = description_override or metadata_description_path
        sample_submission_path = sample_submission_override or metadata_sample_submission_path
        if description_path is None:
            ready_to_execute = False
            warnings.append(
                "competition description metadata is missing; cache-hit plans stay non-runnable until description.md can be staged."
            )
        if sample_submission_path is None:
            ready_to_execute = False
            warnings.append(
                "sample submission metadata is missing; cache-hit plans stay non-runnable until sample_submission.csv can be staged."
            )
    elif resolved_inputs.source_kind == InputSourceKind.LOCAL_ZIP:
        warnings.append(
            "Local zip input is supported in the live adapter; this preview does not enumerate the temporary public/private staging commands."
        )
    elif resolved_inputs.source_kind == InputSourceKind.DATA_DIR:
        safe_public_dir = _safe_public_data_dir(Path(job_spec.mode_spec.data_dir or ""))
        if safe_public_dir is None:
            ready_to_execute = False
            warnings.append(
                "data_dir must point to a public competition directory, a prepared directory, or a competition root containing prepared/public."
            )
        else:
            description_override = _optional_existing_path(job_spec.mode_spec.description_path)
            sample_submission_override = _optional_existing_path(job_spec.mode_spec.sample_submission_path)
            description_path = description_override
            sample_submission_path = sample_submission_override
            if description_path is None and not (safe_public_dir / "description.md").is_file():
                ready_to_execute = False
                warnings.append(
                    "description.md is missing from the safe public data_dir input; provide --description-path before launch."
                )
            if sample_submission_path is None and not (safe_public_dir / "sample_submission.csv").is_file():
                ready_to_execute = False
                warnings.append(
                    "sample_submission.csv is missing from the safe public data_dir input; provide --sample-submission-path before launch."
                )

    time_limit_secs = parse_duration_seconds(job_spec.runtime_profile.time_limit)
    sandbox_env = build_mle_sandbox_env(
        job_id=_preview_job_id(job_spec, workspace),
        objective=job_spec.objective,
    )
    llm_env = build_mle_session_env(
        job_spec.llm_profile,
        time_limit_secs=time_limit_secs,
        competition_id=job_spec.mode_spec.competition_name or "mle-local",
        hardware=_hardware_label(job_spec.runtime_profile.gpu_ids),
    )
    if not any(llm_env.get(key) for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")):
        warnings.append(
            "no LLM API key is set in the host environment; this plan is safe to inspect but not ready for a live run."
        )

    runtime_profile = _shared_runtime_profile(job_spec)
    validation_command: str | None = None
    validation: ValidationPlan | None = None
    if job_spec.runtime_profile.run_final_validation:
        if resolved_inputs.source_kind == InputSourceKind.COMPETITION_NAME and resolved_inputs.cache_prepared_exists:
            try:
                validation = _legacy_grade_plan(
                    competition_name=resolved_inputs.competition_name or "unknown-competition",
                    cache_root=Path(resolved_inputs.cache_root),
                    workspace=workspace,
                )
            except ValueError as exc:
                ready_to_execute = False
                warnings.append(str(exc))
        elif resolved_inputs.source_kind == InputSourceKind.LOCAL_ZIP:
            validation = ValidationPlan(
                kind=ValidationKind.LEGACY_GRADE,
                description="Live adapter prepares local zip input into temporary public/private trees and runs legacy host grading after the solve loop.",
                passthrough_env_keys=[],
            )
        else:
            validation_command = DEFAULT_FORMAT_ONLY_VALIDATION

    preview = preview_shared_mle_runtime(
        job_id=_preview_job_id(job_spec, workspace),
        workspace=workspace,
        runtime_profile=runtime_profile,
        env=sandbox_env,
        docker_binary=docker_binary,
        validation_command=validation_command,
    )
    mounts = [
        _mount_from_session(mount.source, mount.target, read_only=mount.read_only)
        for mount in preview.session_spec.mounts
    ]

    if validation_command:
        validation = _shared_validation_plan(preview)

    if validation and validation.kind == ValidationKind.LEGACY_GRADE and not validation.docker_cleanup_command:
        validation.docker_cleanup_command = list(preview.cleanup_command)

    dockerfile_for_report = (
        str(Path(dockerfile_path).expanduser().resolve())
        if dockerfile_path
        else ""
    )
    image_inspect_command = list(preview.prepare_commands[0]) if preview.prepare_commands else None
    redacted_env = _redact_env(sandbox_env)
    redacted_run_command = _redact_command(list(preview.run_command))
    redacted_exec_command = (
        _redact_command(list(preview.exec_command))
        if preview.exec_command
        else None
    )
    redacted_cleanup_command = _redact_command(list(preview.cleanup_command))
    redacted_validation = validation
    if redacted_validation and redacted_validation.kind == ValidationKind.FORMAT_ONLY:
        redacted_validation = ValidationPlan(
            kind=redacted_validation.kind,
            description=redacted_validation.description,
            passthrough_env_keys=list(redacted_validation.passthrough_env_keys),
            cwd=redacted_validation.cwd,
            env=_redact_env(dict(redacted_validation.env)),
            host_command=list(redacted_validation.host_command) if redacted_validation.host_command else None,
            docker_run_command=_redact_command(list(redacted_validation.docker_run_command or [])),
            docker_exec_command=_redact_command(list(redacted_validation.docker_exec_command or [])),
            docker_cleanup_command=_redact_command(list(redacted_validation.docker_cleanup_command or [])),
        )

    return RuntimeOrchestrationPlan(
        ready_to_execute=ready_to_execute,
        build_policy=policy,
        docker_binary=docker_binary,
        dockerfile_path=dockerfile_for_report,
        image_tag=preview.image_tag,
        container_name=_command_container_name(preview.run_command),
        workspace=workspace,
        env=redacted_env,
        passthrough_env_keys=present_proxy_env_keys(),
        mounts=mounts,
        host_setup_commands=_host_setup_commands(
            workspace,
            job_spec=job_spec,
            resolved_inputs=resolved_inputs,
            description_path=description_path,
            sample_submission_path=sample_submission_path,
        ),
        prepare_plan=prepare_plan,
        image_inspect_command=image_inspect_command,
        build_command=None,
        docker_run_command=redacted_run_command,
        docker_exec_command=redacted_exec_command,
        docker_cleanup_command=redacted_cleanup_command,
        validation=redacted_validation,
        warnings=warnings,
    )
