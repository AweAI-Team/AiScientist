from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import PurePosixPath
from pathlib import Path

from aisci_core.models import (
    JobPaths,
    NetworkPolicy,
    PullPolicy,
    RuntimeProfile,
    UIDGIDMode,
    ValidationReport,
    WorkspaceLayout,
)
from aisci_runtime_docker.models import (
    ContainerSession,
    DockerExecutionResult,
    DockerProfile,
    SessionMount,
    SessionSpec,
)


class DockerRuntimeError(RuntimeError):
    pass


class AgentSessionManager:
    def __init__(self):
        self._docker = shutil.which("docker") or "docker"

    def can_use_docker(self) -> bool:
        if shutil.which(self._docker) is None and shutil.which("docker") is None:
            return False
        try:
            return self._run([self._docker, "info"], check=False).exit_code == 0
        except Exception:
            return False

    def image_exists(self, image_ref: str) -> bool:
        return self._run([self._docker, "image", "inspect", image_ref], check=False).exit_code == 0

    def pull_image(self, image_ref: str) -> None:
        self._run([self._docker, "pull", image_ref])

    def prepare_image(self, profile: DockerProfile, runtime_profile: RuntimeProfile) -> str:
        image_ref = (runtime_profile.image or "").strip() or profile.image
        if not image_ref:
            raise DockerRuntimeError(
                "No runtime image was configured. Pass --image or configure config/image_profiles.yaml."
            )

        pull_policy = runtime_profile.pull_policy or profile.pull_policy
        if pull_policy == PullPolicy.ALWAYS:
            self.pull_image(image_ref)
        elif pull_policy == PullPolicy.IF_MISSING:
            if not self.image_exists(image_ref):
                self.pull_image(image_ref)
        elif pull_policy == PullPolicy.NEVER:
            if not self.image_exists(image_ref):
                raise DockerRuntimeError(
                    f"Runtime image is not present locally and pull_policy=never: {image_ref}"
                )
        else:
            raise DockerRuntimeError(f"Unsupported pull policy: {pull_policy}")

        return image_ref

    def ensure_layout(self, job_paths: JobPaths, layout: WorkspaceLayout) -> None:
        for path in self._layout_paths(job_paths, layout).values():
            path.mkdir(parents=True, exist_ok=True)

    def create_session_spec(
        self,
        job_id: str,
        job_paths: JobPaths,
        profile: DockerProfile,
        runtime_profile: RuntimeProfile,
        *,
        layout: WorkspaceLayout,
        entry_command: list[str] | tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        extra_mounts: tuple[SessionMount, ...] | list[SessionMount] = (),
    ) -> SessionSpec:
        mounts = tuple([*self._layout_mounts(job_paths, layout), *list(extra_mounts)])
        return SessionSpec(
            job_id=job_id,
            workspace_layout=layout,
            profile=profile,
            runtime_profile=runtime_profile,
            mounts=mounts,
            workdir=workdir or self._default_workdir(layout),
            entry_command=tuple(entry_command),
            env=tuple(sorted((env or {}).items())),
            labels=tuple(sorted(self._labels_for(job_id, layout, profile).items())),
            run_as_user=self._run_as_user(runtime_profile),
        )

    def start_session(self, spec: SessionSpec, image_tag: str) -> ContainerSession:
        container_name = f"aisci-{spec.job_id}-{secrets.token_hex(4)}"
        command = [
            self._docker,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            spec.workdir,
            *self._network_args(spec.runtime_profile.network_policy),
            *self._gpu_args(spec.runtime_profile),
            *self._resource_args(spec.runtime_profile),
            *self._user_args(spec.run_as_user),
        ]
        for key, value in spec.labels:
            command.extend(["--label", f"{key}={value}"])
        for mount in spec.mounts:
            source = str(mount.source.resolve())
            target = mount.target
            suffix = ":ro" if mount.read_only else ""
            command.extend(["-v", f"{source}:{target}{suffix}"])
        for key, value in spec.env:
            command.extend(["-e", f"{key}={value}"])
        command.append(image_tag)
        command.extend(spec.entry_command or tuple(spec.profile.default_command))
        self._run(command)
        return ContainerSession(
            container_name=container_name,
            image_tag=image_tag,
            profile=spec.profile,
            runtime_profile=spec.runtime_profile,
            workspace_layout=spec.workspace_layout,
            mounts=spec.mounts,
            workdir=spec.workdir,
            labels=spec.labels,
            run_as_user=spec.run_as_user,
            started_at=datetime.now(timezone.utc),
        )

    def exec(
        self,
        session: ContainerSession,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        check: bool = False,
        timeout_seconds: int | None = None,
    ) -> DockerExecutionResult:
        cmd = [self._docker, "exec", "-w", workdir or session.workdir]
        for key, value in (env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([session.container_name, "bash", "-lc", command])
        return self._run(cmd, check=check, timeout=timeout_seconds)

    def copy_to_session(
        self,
        session: ContainerSession,
        source: Path,
        destination: str,
    ) -> None:
        parent = str(PurePosixPath(destination).parent)
        self.exec(
            session,
            f"mkdir -p {json.dumps(parent)}",
            workdir=session.workdir,
            check=True,
            timeout_seconds=30,
        )
        self._run(
            [self._docker, "cp", str(source), f"{session.container_name}:{destination}"],
            timeout=60,
        )

    def copy_from_session(
        self,
        session: ContainerSession,
        source: str,
        destination: Path,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [self._docker, "cp", f"{session.container_name}:{source}", str(destination)],
            timeout=60,
        )

    def run_validation(
        self,
        spec: SessionSpec,
        image_tag: str,
        validation_command: str,
        *,
        workdir: str | None = None,
    ) -> ValidationReport:
        started = ValidationReport(
            status="skipped",
            summary="placeholder",
            runtime_profile_hash=self._runtime_hash(spec.runtime_profile),
            container_image=image_tag,
        ).started_at
        session = self.start_session(spec, image_tag)
        try:
            result = self.exec(session, validation_command, workdir=workdir, check=False)
            status = "passed" if result.exit_code == 0 else "failed"
            summary = (
                "Validation completed successfully."
                if result.exit_code == 0
                else "Validation command failed."
            )
            return ValidationReport(
                status=status,
                summary=summary,
                details={
                    "command": validation_command,
                    "exit_code": result.exit_code,
                    "output": result.combined_output,
                },
                runtime_profile_hash=self._runtime_hash(spec.runtime_profile),
                container_image=image_tag,
                started_at=started,
            )
        finally:
            self.cleanup(session)

    def cleanup(self, session: ContainerSession) -> None:
        self._run([self._docker, "rm", "-f", session.container_name], check=False)

    def inspect_session(self, session: ContainerSession) -> dict[str, object]:
        result = self._run([self._docker, "inspect", session.container_name], check=False, timeout=30)
        if result.exit_code != 0 or not result.stdout:
            return {}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, list) or not payload:
            return {}
        record = payload[0]
        if not isinstance(record, dict):
            return {}
        config = record.get("Config") if isinstance(record.get("Config"), dict) else {}
        host_config = record.get("HostConfig") if isinstance(record.get("HostConfig"), dict) else {}
        state = record.get("State") if isinstance(record.get("State"), dict) else {}
        return {
            "name": record.get("Name"),
            "image": record.get("Image"),
            "user": config.get("User"),
            "working_dir": config.get("WorkingDir"),
            "labels": config.get("Labels") if isinstance(config.get("Labels"), dict) else {},
            "network_mode": host_config.get("NetworkMode"),
            "memory_limit": host_config.get("Memory"),
            "nano_cpus": host_config.get("NanoCpus"),
            "running": state.get("Running"),
            "status": state.get("Status"),
        }

    def collect_artifacts(self, job_paths: JobPaths) -> list[Path]:
        return [path for path in sorted(job_paths.artifacts_dir.rglob("*")) if path.is_file()]

    def _layout_paths(self, job_paths: JobPaths, layout: WorkspaceLayout) -> dict[str, Path]:
        if layout == WorkspaceLayout.PAPER:
            return {
                "home": job_paths.workspace_dir,
                "paper": job_paths.workspace_dir / "paper",
                "submission": job_paths.workspace_dir / "submission",
                "agent": job_paths.workspace_dir / "agent",
            }
        return {
            "home": job_paths.workspace_dir,
            "data": job_paths.workspace_dir / "data",
            "code": job_paths.workspace_dir / "code",
            "submission": job_paths.workspace_dir / "submission",
            "agent": job_paths.workspace_dir / "agent",
        }

    def _layout_mounts(self, job_paths: JobPaths, layout: WorkspaceLayout) -> list[SessionMount]:
        paths = self._layout_paths(job_paths, layout)
        return [
            SessionMount(paths["home"], "/home"),
            SessionMount(job_paths.logs_dir, "/home/logs"),
            SessionMount(job_paths.logs_dir, "/workspace/logs"),
        ]

    def _default_workdir(self, layout: WorkspaceLayout) -> str:
        if layout == WorkspaceLayout.PAPER:
            return "/home/submission"
        return "/home/code"

    def _run(
        self,
        command: list[str],
        check: bool = True,
        timeout: int | None = None,
    ) -> DockerExecutionResult:
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            result = DockerExecutionResult(
                command=command,
                exit_code=137,
                stdout="",
                stderr=f"Docker command timed out after {timeout}s.",
            )
            if check:
                raise DockerRuntimeError(result.combined_output or f"Docker command failed: {' '.join(command)}")
            return result
        result = DockerExecutionResult(
            command=command,
            exit_code=completed.returncode,
            stdout=(completed.stdout or "").strip(),
            stderr=(completed.stderr or "").strip(),
        )
        if check and result.exit_code != 0:
            raise DockerRuntimeError(
                result.combined_output or f"Docker command failed: {' '.join(command)}"
            )
        return result

    def _network_args(self, policy: NetworkPolicy) -> list[str]:
        if policy == NetworkPolicy.HOST:
            return ["--network", "host"]
        if policy == NetworkPolicy.NONE:
            return ["--network", "none"]
        return []

    def _resource_args(self, runtime_profile: RuntimeProfile) -> list[str]:
        args: list[str] = []
        if runtime_profile.cpu_limit:
            args.extend(["--cpus", str(runtime_profile.cpu_limit)])
        elif runtime_profile.nano_cpus is not None:
            whole_cpus, fractional_nanos = divmod(runtime_profile.nano_cpus, 1_000_000_000)
            cpu_limit = str(whole_cpus)
            if fractional_nanos:
                cpu_limit = f"{whole_cpus}.{fractional_nanos:09d}".rstrip("0")
            args.extend(["--cpus", cpu_limit])
        if runtime_profile.memory_limit:
            args.extend(["--memory", str(runtime_profile.memory_limit)])
        if runtime_profile.shm_size:
            args.extend(["--shm-size", str(runtime_profile.shm_size)])
        return args

    def _gpu_args(self, runtime_profile: RuntimeProfile) -> list[str]:
        if runtime_profile.gpu_ids:
            return ["--gpus", f"device={','.join(runtime_profile.gpu_ids)}"]
        if runtime_profile.gpu_count > 0:
            return ["--gpus", str(runtime_profile.gpu_count)]
        return []

    def _user_args(self, run_as_user: str | None) -> list[str]:
        if not run_as_user:
            return []
        return ["--user", run_as_user]

    def _run_as_user(self, runtime_profile: RuntimeProfile) -> str | None:
        if runtime_profile.uid_gid_mode != UIDGIDMode.HOST:
            return None
        if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
            return None
        return f"{os.getuid()}:{os.getgid()}"

    def _labels_for(
        self,
        job_id: str,
        layout: WorkspaceLayout,
        profile: DockerProfile,
    ) -> dict[str, str]:
        return {
            "aisci.job_id": job_id,
            "aisci.workspace_layout": layout.value,
            "aisci.image_profile": profile.name,
        }

    def _runtime_hash(self, runtime_profile: RuntimeProfile) -> str:
        return hashlib.sha256(
            json.dumps(runtime_profile.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
