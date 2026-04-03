from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any, Literal

from aisci_core.models import NetworkPolicy


class InputSourceKind(StrEnum):
    COMPETITION_NAME = "competition_name"
    LOCAL_ZIP = "local_zip"
    DATA_DIR = "data_dir"
    WORKSPACE_BUNDLE = "workspace_bundle"
    COMPETITION_BUNDLE = "competition_bundle"


class DryRunStatus(StrEnum):
    READY = "ready"
    NEEDS_PREPARE = "needs_prepare"


class DockerBuildPolicy(StrEnum):
    AUTO = "auto"
    FORCE = "force"
    NEVER = "never"


class ValidationKind(StrEnum):
    NONE = "none"
    FORMAT_ONLY = "format_only"
    LEGACY_GRADE = "legacy_grade"


def _json_safe(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class MLEPhase1ModeSpec:
    competition_name: str | None = None
    competition_zip_path: str | None = None
    mlebench_data_dir: str | None = None

    workspace_bundle_zip: str | None = None
    competition_bundle_zip: str | None = None
    data_dir: str | None = None
    code_repo_zip: str | None = None
    description_path: str | None = None
    sample_submission_path: str | None = None
    validation_command: str | None = None
    grading_config_path: str | None = None
    metric_direction: Literal["maximize", "minimize"] | None = None
    evaluation_contract_summary: str | None = None
    expected_output_format: str | None = None

    def __post_init__(self) -> None:
        if not any(
            [
                self.competition_name,
                self.competition_zip_path,
                self.workspace_bundle_zip,
                self.competition_bundle_zip,
                self.data_dir,
            ]
        ):
            raise ValueError(
                "phase1 mle job requires competition_name, competition_zip_path, "
                "workspace_bundle_zip, competition_bundle_zip, or data_dir"
            )
        if self.metric_direction not in {None, "maximize", "minimize"}:
            raise ValueError("metric_direction must be 'maximize', 'minimize', or omitted")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class MLEPhase1RuntimeSpec:
    gpu_ids: list[str] = field(default_factory=list)
    gpu_count: int = 0
    time_limit: str = "24h"
    dockerfile_path: str | None = None
    run_final_validation: bool = False
    network_policy: NetworkPolicy = NetworkPolicy.BRIDGE
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.gpu_count = len(self.gpu_ids)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class MLEPhase1JobSpec:
    runtime_profile: MLEPhase1RuntimeSpec
    mode_spec: MLEPhase1ModeSpec
    objective: str = "mle optimization job"
    llm_profile: str = "gpt-5.4"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class LegacyPreparePlan:
    competition_name: str
    data_dir: str
    python_executable: str
    module: str
    cwd: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class RuntimeWorkspaceLayout:
    run_root: str
    input_dir: str
    workspace_dir: str
    data_dir: str
    code_dir: str
    submission_dir: str
    agent_dir: str
    workspace_logs_dir: str
    host_logs_dir: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class DockerMountPlan:
    source: str
    target: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class ValidationPlan:
    kind: ValidationKind
    description: str
    passthrough_env_keys: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    host_command: list[str] | None = None
    docker_run_command: list[str] | None = None
    docker_exec_command: list[str] | None = None
    docker_cleanup_command: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class RuntimeOrchestrationPlan:
    ready_to_execute: bool
    build_policy: DockerBuildPolicy
    docker_binary: str
    dockerfile_path: str
    image_tag: str
    container_name: str
    workspace: RuntimeWorkspaceLayout
    env: dict[str, str] = field(default_factory=dict)
    passthrough_env_keys: list[str] = field(default_factory=list)
    mounts: list[DockerMountPlan] = field(default_factory=list)
    host_setup_commands: list[list[str]] = field(default_factory=list)
    prepare_plan: LegacyPreparePlan | None = None
    image_inspect_command: list[str] | None = None
    build_command: list[str] | None = None
    docker_run_command: list[str] | None = None
    docker_exec_command: list[str] | None = None
    docker_cleanup_command: list[str] | None = None
    validation: ValidationPlan | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class ResolvedInputState:
    source_kind: InputSourceKind
    cache_root: str
    competition_name: str | None = None
    competition_zip_path: str | None = None
    zip_exists: bool = False
    cache_dir: str | None = None
    cache_dir_exists: bool = False
    cache_prepared_dir: str | None = None
    cache_prepared_exists: bool = False
    legacy_prepare_plan: LegacyPreparePlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class Phase1DryRunReport:
    status: DryRunStatus
    wait_requested: bool
    api_key_required: bool
    api_key_present: bool
    job_spec: MLEPhase1JobSpec
    resolved_inputs: ResolvedInputState
    warnings: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)
