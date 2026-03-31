from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

from aisci_agent_runtime.llm_client import LLMConfig, create_llm_client
from aisci_agent_runtime.llm_profiles import (
    backend_env_values,
    missing_backend_env_vars,
    resolve_llm_profile,
)
from aisci_agent_runtime.trace import AgentTraceWriter
from aisci_core.logging_utils import append_log
from aisci_core.models import ArtifactRecord, JobRecord, PaperSpec, RunPhase, ValidationReport, WorkspaceLayout
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_domain_paper.engine import EmbeddedPaperEngine, PaperRuntimeConfig
from aisci_runtime_docker.profiles import default_paper_profile
from aisci_runtime_docker.runtime import DockerRuntimeError, DockerRuntimeManager
from aisci_runtime_docker.shell_interface import DockerShellInterface


class PaperDomainAdapter:
    def __init__(self, runtime: DockerRuntimeManager):
        self.runtime = runtime

    def run(self, job: JobRecord) -> dict[str, object]:
        assert isinstance(job.mode_spec, PaperSpec)
        job_paths = ensure_job_dirs(resolve_job_paths(job.id))
        append_log(job_paths.logs_dir / "job.log", "paper job started")
        self.runtime.ensure_layout(job_paths, WorkspaceLayout.PAPER)
        self._stage_inputs(job.mode_spec, job_paths)

        submission_dir = job_paths.workspace_dir / "submission"
        if not (submission_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=submission_dir, check=False, capture_output=True)

        profile = resolve_llm_profile(job.llm_profile, default_for="paper")
        self._ensure_runtime_ready(job, job_paths, profile)

        docker_profile = default_paper_profile(job.runtime_profile.image_profile)
        image_tag = self.runtime.prepare_image(docker_profile, job.runtime_profile)
        spec = self.runtime.create_session_spec(
            job.id,
            job_paths,
            docker_profile,
            job.runtime_profile,
            layout=WorkspaceLayout.PAPER,
            workdir="/home/submission",
            env=self._sandbox_env(job),
        )
        session = self.runtime.start_session(spec, image_tag)
        self._write_session_info(job_paths, session, profile.name)

        keep_session = False
        try:
            self._run_real_loop(job, job_paths, session, profile)
        except Exception:
            if job.runtime_profile.keep_container_on_failure:
                keep_session = True
                append_log(
                    job_paths.logs_dir / "job.log",
                    f"paper sandbox preserved for debugging: {session.container_name}",
                )
                self._write_session_info(job_paths, session, profile.name, kept_on_failure=True)
            raise
        finally:
            if not keep_session:
                self.runtime.cleanup(session)

        artifacts = self._collect_artifacts(job_paths)
        validation = self._maybe_validate(job, job_paths, image_tag=image_tag)
        return {"artifacts": artifacts, "validation_report": validation}

    def _ensure_runtime_ready(self, job: JobRecord, job_paths, profile) -> None:
        if not self.runtime.can_use_docker():
            message = "Paper mode requires a reachable Docker daemon. No local fallback loop is available."
            append_log(job_paths.logs_dir / "job.log", message)
            raise DockerRuntimeError(message)
        missing = missing_backend_env_vars(profile)
        if missing:
            message = (
                f"Paper mode requires backend credentials for profile {profile.name}: "
                f"{', '.join(missing)}. No local fallback loop is available."
            )
            append_log(job_paths.logs_dir / "job.log", message)
            raise RuntimeError(message)
        append_log(
            job_paths.logs_dir / "job.log",
            f"resolved llm profile={profile.name} backend={profile.backend_name} provider={profile.provider} model={profile.model} api={profile.api_mode}",
        )

    def _stage_inputs(self, spec: PaperSpec, job_paths) -> None:
        paper_dir = job_paths.workspace_dir / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)

        if spec.paper_bundle_zip:
            archive = Path(spec.paper_bundle_zip).resolve()
            shutil.copy2(archive, job_paths.input_dir / "paper_bundle.zip")
            self._extract_zip(archive, paper_dir)
        if spec.context_bundle_zip:
            archive = Path(spec.context_bundle_zip).resolve()
            shutil.copy2(archive, job_paths.input_dir / "context_bundle.zip")
            self._extract_zip(archive, paper_dir)
        if spec.pdf_path:
            src = Path(spec.pdf_path).resolve()
            shutil.copy2(src, paper_dir / "paper.pdf")
            shutil.copy2(src, job_paths.input_dir / "paper.pdf")
        if spec.paper_md_path:
            src = Path(spec.paper_md_path).resolve()
            shutil.copy2(src, paper_dir / "paper.md")
        self._copy_optional(spec.rubric_path, paper_dir / "rubric.json")
        self._copy_optional(spec.blacklist_path, paper_dir / "blacklist.txt")
        self._copy_optional(spec.addendum_path, paper_dir / "addendum.md")
        for material in spec.supporting_materials:
            src = Path(material).resolve()
            if src.is_file():
                shutil.copy2(src, paper_dir / src.name)
        if spec.submission_seed_repo_zip:
            self._extract_zip(Path(spec.submission_seed_repo_zip).resolve(), job_paths.workspace_dir / "submission")

    def _extract_zip(self, archive_path: Path, destination: Path) -> None:
        with ZipFile(archive_path) as zf:
            zf.extractall(destination)

    def _copy_optional(self, source: str | None, destination: Path) -> None:
        if source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source).resolve(), destination)

    def _run_real_loop(self, job: JobRecord, job_paths, session, profile) -> None:
        shell = DockerShellInterface(self.runtime, session, working_dir="/home/submission")
        llm = self._build_llm_client(profile, enable_online_research=job.mode_spec.enable_online_research)
        self._write_resolved_llm_config(job_paths, profile, job.mode_spec.enable_online_research)
        trace = AgentTraceWriter(job_paths.logs_dir)
        engine = EmbeddedPaperEngine(
            config=PaperRuntimeConfig(
                job_id=job.id,
                objective=job.objective,
                llm_profile_name=profile.name,
                time_limit_seconds=self._parse_duration(job.runtime_profile.time_limit),
                max_steps=int(os.environ.get("AISCI_MAX_STEPS", "80")),
                reminder_freq=int(os.environ.get("AISCI_REMINDER_FREQ", "5")),
                enable_online_research=job.mode_spec.enable_online_research,
            ),
            shell=shell,
            llm=llm,
            paper_dir=job_paths.workspace_dir / "paper",
            submission_dir=job_paths.workspace_dir / "submission",
            agent_dir=job_paths.workspace_dir / "agent",
            logs_dir=job_paths.logs_dir,
            state_dir=job_paths.state_dir,
            trace=trace,
        )
        summary = engine.run()
        append_log(job_paths.logs_dir / "job.log", summary)

    def _build_llm_client(self, profile, *, enable_online_research: bool):
        backend_values = backend_env_values(profile)
        return create_llm_client(
            LLMConfig(
                provider=profile.provider,
                model=profile.model,
                api_mode=profile.api_mode,
                max_tokens=profile.max_tokens,
                reasoning_effort=profile.reasoning_effort,
                reasoning_summary=profile.reasoning_summary,
                web_search=profile.web_search and enable_online_research and profile.api_mode == "responses",
                context_window=profile.context_window,
                use_phase=profile.use_phase,
                temperature=profile.temperature,
                clear_thinking=profile.clear_thinking,
                api_key=backend_values.get("api_key"),
                base_url=backend_values.get("base_url"),
                azure_endpoint=backend_values.get("endpoint") or backend_values.get("azure_endpoint"),
                api_version=backend_values.get("api_version"),
                organization=backend_values.get("organization") or backend_values.get("org_id"),
                project=backend_values.get("project"),
            )
        )

    def _sandbox_env(self, job: JobRecord) -> dict[str, str]:
        return {
            "AISCI_JOB_ID": job.id,
            "AISCI_OBJECTIVE": job.objective,
            "LOGS_DIR": "/home/logs",
        }

    def _write_resolved_llm_config(self, job_paths, profile, enable_online_research: bool) -> None:
        backend_values = backend_env_values(profile)
        payload = {
            "profile": profile.name,
            "backend": profile.backend_name,
            "provider": profile.provider,
            "model": profile.model,
            "api_mode": profile.api_mode,
            "limits": {
                "max_completion_tokens": profile.max_tokens,
                "context_window": profile.context_window,
            },
            "reasoning": {
                "effort": profile.reasoning_effort,
                "summary": profile.reasoning_summary,
            },
            "features": {
                "web_search_requested": enable_online_research,
                "web_search_enabled": profile.web_search and enable_online_research and profile.api_mode == "responses",
                "use_phase": profile.use_phase,
                "clear_thinking": profile.clear_thinking,
            },
            "sampling": {
                "temperature": profile.temperature,
            },
            "backend_env": {
                name: {"present": bool(value)}
                for name, value in sorted(backend_values.items())
            },
            "backend_env_vars": {
                name: spec.env_var
                for name, spec in sorted((profile.backend_env or {}).items())
            },
        }
        destination = job_paths.state_dir / "resolved_llm_config.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_session_info(self, job_paths, session, llm_profile: str, *, kept_on_failure: bool = False) -> None:
        inspect_summary = self.runtime.inspect_session(session)
        payload = {
            "container_name": session.container_name,
            "image_ref": session.image_tag,
            "runtime_image_profile": session.profile.name,
            "pull_policy": session.runtime_profile.pull_policy.value if session.runtime_profile.pull_policy else session.profile.pull_policy.value,
            "workdir": session.workdir,
            "run_as_user": session.run_as_user,
            "started_at": session.started_at.isoformat(),
            "llm_profile": llm_profile,
            "kept_on_failure": kept_on_failure,
            "labels": {key: value for key, value in session.labels},
            "mounts": [
                {
                    "source": str(mount.source),
                    "target": mount.target,
                    "read_only": mount.read_only,
                }
                for mount in session.mounts
            ],
            "docker_inspect": inspect_summary,
        }
        destination = job_paths.state_dir / "sandbox_session.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _collect_artifacts(self, job_paths) -> list[ArtifactRecord]:
        known: list[tuple[str, Path, RunPhase]] = [
            ("paper_analysis", job_paths.workspace_dir / "agent" / "paper_analysis" / "summary.md", RunPhase.ANALYZE),
            ("paper_structure", job_paths.workspace_dir / "agent" / "paper_analysis" / "structure.md", RunPhase.ANALYZE),
            ("paper_algorithm", job_paths.workspace_dir / "agent" / "paper_analysis" / "algorithm.md", RunPhase.ANALYZE),
            ("paper_experiments", job_paths.workspace_dir / "agent" / "paper_analysis" / "experiments.md", RunPhase.ANALYZE),
            ("paper_baseline", job_paths.workspace_dir / "agent" / "paper_analysis" / "baseline.md", RunPhase.ANALYZE),
            ("prioritized_tasks", job_paths.workspace_dir / "agent" / "prioritized_tasks.md", RunPhase.PRIORITIZE),
            ("plan", job_paths.workspace_dir / "agent" / "plan.md", RunPhase.IMPLEMENT),
            ("impl_log", job_paths.workspace_dir / "agent" / "impl_log.md", RunPhase.IMPLEMENT),
            ("exp_log", job_paths.workspace_dir / "agent" / "exp_log.md", RunPhase.VALIDATE),
            ("reproduce_script", job_paths.workspace_dir / "submission" / "reproduce.sh", RunPhase.FINALIZE),
            ("capabilities", job_paths.state_dir / "capabilities.json", RunPhase.ANALYZE),
            ("resolved_llm_config", job_paths.state_dir / "resolved_llm_config.json", RunPhase.INGEST),
            ("self_check_report", job_paths.workspace_dir / "agent" / "final_self_check.md", RunPhase.VALIDATE),
            ("self_check_report_json", job_paths.workspace_dir / "agent" / "final_self_check.json", RunPhase.VALIDATE),
            ("prompt", job_paths.state_dir / "paper_main_prompt.md", RunPhase.INGEST),
            ("agent_log", job_paths.logs_dir / "agent.log", RunPhase.FINALIZE),
            ("conversation_log", job_paths.logs_dir / "conversation.jsonl", RunPhase.FINALIZE),
            ("orchestrator_state", job_paths.logs_dir / "paper_session_state.json", RunPhase.FINALIZE),
            ("sandbox_session", job_paths.state_dir / "sandbox_session.json", RunPhase.FINALIZE),
        ]
        artifacts: list[ArtifactRecord] = []
        for artifact_type, path, phase in known:
            if path.exists():
                artifacts.append(
                    ArtifactRecord(
                        artifact_type=artifact_type,
                        path=str(path),
                        phase=phase,
                        size_bytes=path.stat().st_size,
                        metadata={},
                    )
                )
        subagent_logs = job_paths.logs_dir / "subagent_logs"
        if subagent_logs.exists():
            for path in sorted(subagent_logs.rglob("*")):
                if path.is_file():
                    artifacts.append(
                        ArtifactRecord(
                            artifact_type="subagent_log",
                            path=str(path),
                            phase=RunPhase.FINALIZE,
                            size_bytes=path.stat().st_size,
                            metadata={},
                        )
                    )
        return artifacts

    def _maybe_validate(self, job: JobRecord, job_paths, *, image_tag: str | None = None) -> ValidationReport:
        if not job.runtime_profile.run_final_validation:
            return ValidationReport(
                status="skipped",
                summary="Final validation disabled for this job.",
                runtime_profile_hash="disabled",
                container_image="not-built",
            )
        if not self.runtime.can_use_docker():
            return ValidationReport(
                status="failed",
                summary="Docker is not available on this machine.",
                details={"reason": "docker_not_found"},
                runtime_profile_hash="docker-missing",
                container_image="not-built",
            )
        reproduce_path = job_paths.workspace_dir / "submission" / "reproduce.sh"
        if not reproduce_path.exists():
            return ValidationReport(
                status="failed",
                summary="No reproduce.sh was staged for validation.",
                details={"reason": "reproduce_missing"},
                runtime_profile_hash="reproduce-missing",
                container_image="not-built",
            )
        try:
            profile = default_paper_profile(job.runtime_profile.image_profile)
            built_image = image_tag or self.runtime.prepare_image(profile, job.runtime_profile)
            spec = self.runtime.create_session_spec(
                job.id,
                job_paths,
                profile,
                job.runtime_profile,
                layout=WorkspaceLayout.PAPER,
                workdir="/home/submission",
            )
            return self.runtime.run_validation(
                spec,
                built_image,
                "bash reproduce.sh",
                workdir="/home/submission",
            )
        except DockerRuntimeError as exc:
            return ValidationReport(
                status="failed",
                summary="Final validation failed while preparing the runtime image.",
                details={"reason": str(exc)},
                runtime_profile_hash="runtime-image-failed",
                container_image="image-prepare-failed",
            )

    def _parse_duration(self, text: str) -> int:
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
        return total or 24 * 3600
