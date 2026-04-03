from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from aisci_agent_runtime.llm_profiles import llm_env
from aisci_core.models import JobPaths, RuntimeProfile, ValidationReport, WorkspaceLayout
from aisci_runtime_docker.models import DockerExecutionResult, SessionMount, SessionSpec
from aisci_runtime_docker.profiles import default_mle_profile
from aisci_runtime_docker.runtime import DockerRuntimeManager

from aisci_domain_mle.contracts import RuntimeWorkspaceLayout

DEFAULT_CONTEXT_REDUCE_STRATEGY = "summary"
MLE_RUNTIME_REPO_TARGET = "/opt/aisci-src"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
FIXED_PROFILE_ENV: dict[str, dict[str, str]] = {
    "glm-5": {
        "OPENAI_API_VERSION": "2024-02-01",
    },
    "gemini-3-flash": {
        "OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}


def domain_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mle_runtime_repo_target() -> str:
    return MLE_RUNTIME_REPO_TARGET


def mle_runtime_extra_mounts() -> tuple[SessionMount, ...]:
    return (
        SessionMount(domain_repo_root(), mle_runtime_repo_target(), read_only=True),
    )


def domain_llm_profile_file() -> str:
    return str((domain_repo_root() / "config" / "llm_profiles.yaml").resolve())


def resolve_shared_profile_name(profile_name: str | None) -> str | None:
    if profile_name is None:
        return None
    cleaned = profile_name.strip()
    if not cleaned:
        return None
    return cleaned


def shared_llm_env(
    profile_name: str | None,
    *,
    default_for: str | None = None,
) -> dict[str, str]:
    return llm_env(
        resolve_shared_profile_name(profile_name),
        default_for=default_for,
        profile_file=domain_llm_profile_file(),
    )


def default_domain_mle_profile():
    with mock.patch.dict(
        os.environ,
        {"AISCI_REPO_ROOT": str(domain_repo_root())},
        clear=False,
    ):
        return default_mle_profile()


def build_mle_session_env(
    profile_name: str | None,
    *,
    time_limit_secs: int,
    competition_id: str,
    hardware: str,
    stub_env_keys: tuple[str, ...] = (),
) -> dict[str, str]:
    normalized_profile = resolve_shared_profile_name(profile_name)
    env = shared_llm_env(normalized_profile)
    for key, value in FIXED_PROFILE_ENV.get(normalized_profile or "", {}).items():
        env[key] = value
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    for key in stub_env_keys:
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.update(
        {
            "TIME_LIMIT_SECS": str(time_limit_secs),
            "LOGS_DIR": "/home/logs",
            "COMPETITION_ID": competition_id,
            "HARDWARE": hardware,
            "AISCI_CONTEXT_REDUCE_STRATEGY": DEFAULT_CONTEXT_REDUCE_STRATEGY,
            "AISCI_REPO_ROOT": mle_runtime_repo_target(),
            "PYTHONPATH": _merge_pythonpath(
                f"{mle_runtime_repo_target()}/src",
                env.get("PYTHONPATH"),
            ),
        }
    )
    return env


def _merge_pythonpath(prefix: str, existing: str | None) -> str:
    parts = [prefix]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def present_proxy_env_keys() -> list[str]:
    return [key for key in PROXY_ENV_KEYS if os.environ.get(key)]


def build_mle_sandbox_env(
    *,
    job_id: str,
    objective: str,
    stub_env_keys: tuple[str, ...] = (),
) -> dict[str, str]:
    env = {
        "AISCI_JOB_ID": job_id,
        "AISCI_OBJECTIVE": objective,
        "LOGS_DIR": "/home/logs",
    }
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    for key in stub_env_keys:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def runtime_workspace_to_job_paths(workspace: RuntimeWorkspaceLayout) -> JobPaths:
    run_root = Path(workspace.run_root).resolve()
    return JobPaths(
        root=run_root,
        input_dir=Path(workspace.input_dir).resolve(),
        workspace_dir=Path(workspace.workspace_dir).resolve(),
        logs_dir=Path(workspace.host_logs_dir).resolve(),
        artifacts_dir=run_root / "artifacts",
        export_dir=run_root / "export",
        state_dir=run_root / "state",
    )


@dataclass(frozen=True)
class SharedRuntimePreview:
    image_tag: str
    session_spec: SessionSpec
    prepare_commands: tuple[list[str], ...]
    run_command: list[str]
    exec_command: list[str] | None
    cleanup_command: list[str]
    validation_report: ValidationReport | None = None
    validation_run_command: list[str] | None = None
    validation_exec_command: list[str] | None = None
    validation_cleanup_command: list[str] | None = None


class RecordingDockerRuntimeManager(DockerRuntimeManager):
    def __init__(self, *, docker_binary: str = "docker"):
        super().__init__()
        self._docker = docker_binary
        self.recorded_results: list[DockerExecutionResult] = []

    def _run(
        self,
        command: list[str],
        check: bool = True,
        timeout: int | None = None,
    ) -> DockerExecutionResult:
        del check, timeout
        result = DockerExecutionResult(
            command=command,
            exit_code=0,
            stdout=self._stdout_for(command),
            stderr="",
        )
        self.recorded_results.append(result)
        return result

    def _stdout_for(self, command: list[str]) -> str:
        if len(command) >= 3 and command[1:3] == ["image", "inspect"]:
            return "[]"
        if len(command) >= 2 and command[1] == "run":
            return "preview-container"
        return ""


def preview_shared_mle_runtime(
    *,
    job_id: str,
    workspace: RuntimeWorkspaceLayout,
    runtime_profile: RuntimeProfile,
    env: dict[str, str],
    docker_binary: str = "docker",
    workdir: str = "/home/code",
    exec_command: str | None = None,
    validation_command: str | None = None,
    extra_mounts: tuple[SessionMount, ...] = (),
) -> SharedRuntimePreview:
    runtime = RecordingDockerRuntimeManager(docker_binary=docker_binary)
    profile = default_domain_mle_profile()
    job_paths = runtime_workspace_to_job_paths(workspace)
    spec = runtime.create_session_spec(
        job_id,
        job_paths,
        profile,
        runtime_profile,
        layout=WorkspaceLayout.MLE,
        workdir=workdir,
        env=env,
        extra_mounts=extra_mounts,
    )

    with mock.patch(
        "aisci_runtime_docker.agent_session.secrets.token_hex",
        side_effect=["preview00", "validate00"],
    ):
        image_tag = runtime.prepare_image(profile, runtime_profile)
        session = runtime.start_session(spec, image_tag)
        if exec_command:
            runtime.exec(session, exec_command, workdir=workdir, check=False)
        runtime.cleanup(session)

        validation_report: ValidationReport | None = None
        if validation_command:
            validation_report = runtime.run_validation(
                spec,
                image_tag,
                validation_command,
                workdir=workdir,
            )

    prepare_commands: list[list[str]] = []
    run_commands: list[list[str]] = []
    exec_commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []
    for result in runtime.recorded_results:
        command = list(result.command)
        if len(command) >= 3 and command[1:3] == ["image", "inspect"]:
            prepare_commands.append(command)
        elif len(command) >= 2 and command[1] == "pull":
            prepare_commands.append(command)
        elif len(command) >= 2 and command[1] == "run":
            run_commands.append(command)
        elif len(command) >= 2 and command[1] == "exec":
            exec_commands.append(command)
        elif len(command) >= 2 and command[1] == "rm":
            cleanup_commands.append(command)

    return SharedRuntimePreview(
        image_tag=image_tag,
        session_spec=spec,
        prepare_commands=tuple(prepare_commands),
        run_command=run_commands[0],
        exec_command=exec_commands[0] if exec_commands else None,
        cleanup_command=cleanup_commands[0],
        validation_report=validation_report,
        validation_run_command=run_commands[1] if len(run_commands) > 1 else None,
        validation_exec_command=exec_commands[1] if len(exec_commands) > 1 else None,
        validation_cleanup_command=cleanup_commands[1] if len(cleanup_commands) > 1 else None,
    )
