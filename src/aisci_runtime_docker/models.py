from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aisci_core.models import PullPolicy, RuntimeProfile, WorkspaceLayout


@dataclass(frozen=True)
class DockerProfile:
    name: str
    image: str
    pull_policy: PullPolicy = PullPolicy.IF_MISSING
    default_command: list[str] = field(default_factory=lambda: ["sleep", "infinity"])


@dataclass(frozen=True)
class SessionMount:
    source: Path
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class SessionSpec:
    job_id: str
    workspace_layout: WorkspaceLayout
    profile: DockerProfile
    runtime_profile: RuntimeProfile
    mounts: tuple[SessionMount, ...]
    workdir: str
    entry_command: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    labels: tuple[tuple[str, str], ...] = ()
    run_as_user: str | None = None


@dataclass(frozen=True)
class ContainerSession:
    container_name: str
    image_tag: str
    profile: DockerProfile
    runtime_profile: RuntimeProfile
    workspace_layout: WorkspaceLayout
    mounts: tuple[SessionMount, ...] = ()
    workdir: str = "/workspace"
    labels: tuple[tuple[str, str], ...] = ()
    run_as_user: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class DockerExecutionResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        if self.stdout and self.stderr:
            return f"{self.stdout}\n{self.stderr}"
        return self.stdout or self.stderr
