from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobType(StrEnum):
    PAPER = "paper"
    MLE = "mle"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunPhase(StrEnum):
    INGEST = "ingest"
    ANALYZE = "analyze"
    PRIORITIZE = "prioritize"
    IMPLEMENT = "implement"
    VALIDATE = "validate"
    FINALIZE = "finalize"


class NetworkPolicy(StrEnum):
    BRIDGE = "bridge"
    HOST = "host"
    NONE = "none"


class WorkspaceLayout(StrEnum):
    PAPER = "paper"
    MLE = "mle"


class UIDGIDMode(StrEnum):
    HOST = "host"
    ROOT = "root"


class ValidationStrategy(StrEnum):
    NONE = "none"
    FRESH_CONTAINER = "fresh_container"
    IN_SESSION = "in_session"


class PullPolicy(StrEnum):
    IF_MISSING = "if-missing"
    ALWAYS = "always"
    NEVER = "never"


class RuntimeProfile(BaseModel):
    gpu_count: int = Field(default=0, ge=0)
    cpu_limit: str | None = None
    memory_limit: str | None = None
    time_limit: str = "24h"
    image: str | None = None
    image_profile: str | None = None
    pull_policy: PullPolicy | None = None
    run_final_validation: bool = False
    network_policy: NetworkPolicy = NetworkPolicy.HOST
    gpu_ids: list[str] = Field(default_factory=list)
    workspace_layout: WorkspaceLayout | None = None
    uid_gid_mode: UIDGIDMode = UIDGIDMode.HOST
    checkpoint_interval_seconds: int = Field(default=300, ge=0)
    validation_strategy: ValidationStrategy = ValidationStrategy.FRESH_CONTAINER
    keep_container_on_failure: bool = False

    @model_validator(mode="after")
    def normalize_validation(self) -> "RuntimeProfile":
        if not self.run_final_validation:
            self.validation_strategy = ValidationStrategy.NONE
        elif self.validation_strategy == ValidationStrategy.NONE:
            self.validation_strategy = ValidationStrategy.FRESH_CONTAINER
        return self


class PaperSpec(BaseModel):
    pdf_path: str | None = None
    paper_bundle_zip: str | None = None
    paper_md_path: str | None = None
    rubric_path: str | None = None
    blacklist_path: str | None = None
    addendum_path: str | None = None
    context_bundle_zip: str | None = None
    supporting_materials: list[str] = Field(default_factory=list)
    submission_seed_repo_zip: str | None = None
    enable_online_research: bool = True

    @model_validator(mode="after")
    def validate_inputs(self) -> "PaperSpec":
        if not any([self.pdf_path, self.paper_bundle_zip, self.paper_md_path]):
            raise ValueError("paper job requires pdf_path, paper_bundle_zip, or paper_md_path")
        return self


class MLESpec(BaseModel):
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

    @model_validator(mode="after")
    def validate_inputs(self) -> "MLESpec":
        if not any([self.workspace_bundle_zip, self.competition_bundle_zip, self.data_dir]):
            raise ValueError(
                "mle job requires workspace_bundle_zip, competition_bundle_zip, or data_dir"
            )
        return self


class JobSpec(BaseModel):
    job_type: JobType
    objective: str
    llm_profile: str
    runtime_profile: RuntimeProfile
    mode_spec: PaperSpec | MLESpec

    @model_validator(mode="after")
    def validate_mode_spec(self) -> "JobSpec":
        if self.job_type == JobType.PAPER and not isinstance(self.mode_spec, PaperSpec):
            raise ValueError("paper job requires PaperSpec")
        if self.job_type == JobType.MLE and not isinstance(self.mode_spec, MLESpec):
            raise ValueError("mle job requires MLESpec")
        if self.runtime_profile.workspace_layout is None:
            self.runtime_profile.workspace_layout = (
                WorkspaceLayout.PAPER if self.job_type == JobType.PAPER else WorkspaceLayout.MLE
            )
        return self


class ArtifactRecord(BaseModel):
    artifact_type: str
    path: str
    phase: RunPhase
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    status: Literal["passed", "failed", "skipped"]
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    runtime_profile_hash: str
    container_image: str
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime = Field(default_factory=utc_now)


class CandidateRecord(BaseModel):
    candidate_path: str
    method_summary: str
    metrics: dict[str, Any] | None = None
    eval_protocol: str
    git_ref: str | None = None
    notes: str | None = None
    is_champion: bool = False


class JobRecord(BaseModel):
    id: str
    job_type: JobType
    status: JobStatus
    phase: RunPhase
    objective: str
    llm_profile: str
    runtime_profile: RuntimeProfile
    mode_spec: PaperSpec | MLESpec
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    worker_pid: int | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if not self.started_at:
            return None
        end = self.ended_at or utc_now()
        return (end - self.started_at).total_seconds()


class EventRecord(BaseModel):
    id: int
    job_id: str
    event_type: str
    phase: RunPhase
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


@dataclass(frozen=True)
class JobPaths:
    root: Path
    input_dir: Path
    workspace_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    export_dir: Path
    state_dir: Path
