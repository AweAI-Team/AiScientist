from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

from aisci_core.logging_utils import append_log
from aisci_core.models import JobRecord, MLESpec, ValidationReport, WorkspaceLayout
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_agent_runtime.llm_profiles import llm_env
from aisci_core.models import ArtifactRecord, RunPhase
from aisci_domain_mle.candidate_registry import CandidateRegistry
from aisci_runtime_docker.profiles import default_mle_profile
from aisci_runtime_docker.runtime import DockerRuntimeError, DockerRuntimeManager


class MLEDomainAdapter:
    def __init__(self, runtime: DockerRuntimeManager):
        self.runtime = runtime

    def run(self, job: JobRecord) -> dict[str, object]:
        assert isinstance(job.mode_spec, MLESpec)
        job_paths = ensure_job_dirs(resolve_job_paths(job.id))
        append_log(job_paths.logs_dir / "job.log", "mle job started")
        self.runtime.ensure_layout(job_paths, WorkspaceLayout.MLE)
        self._stage_inputs(job.mode_spec, job_paths)

        code_dir = job_paths.workspace_dir / "code"
        if not (code_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=code_dir, check=False, capture_output=True)

        self._ensure_runtime_ready(job, job_paths)
        self._run_real_loop(job, job_paths)

        artifacts = self._collect_artifacts(job_paths)
        validation = self._maybe_validate(job, job_paths)
        return {"artifacts": artifacts, "validation_report": validation}

    def _ensure_runtime_ready(self, job: JobRecord, job_paths) -> None:
        if not self.runtime.can_use_docker():
            message = "MLE mode requires a reachable Docker daemon. No local fallback loop is available."
            append_log(job_paths.logs_dir / "job.log", message)
            raise DockerRuntimeError(message)
        env = llm_env(job.llm_profile)
        if not any(env.get(key) for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")):
            message = (
                "MLE mode requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY in the host environment. "
                "No local fallback loop is available."
            )
            append_log(job_paths.logs_dir / "job.log", message)
            raise RuntimeError(message)

    def _stage_inputs(self, spec: MLESpec, job_paths) -> None:
        data_dir = job_paths.workspace_dir / "data"
        code_dir = job_paths.workspace_dir / "code"
        data_dir.mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)

        if spec.workspace_bundle_zip:
            bundle = Path(spec.workspace_bundle_zip).resolve()
            shutil.copy2(bundle, job_paths.input_dir / "workspace_bundle.zip")
            self._extract_zip(bundle, data_dir)
        if spec.competition_bundle_zip:
            bundle = Path(spec.competition_bundle_zip).resolve()
            shutil.copy2(bundle, job_paths.input_dir / "competition_bundle.zip")
            self._extract_zip(bundle, data_dir)
        if spec.code_repo_zip:
            bundle = Path(spec.code_repo_zip).resolve()
            shutil.copy2(bundle, job_paths.input_dir / "code_repo.zip")
            self._extract_zip(bundle, code_dir)
        if spec.data_dir:
            self._copy_tree(Path(spec.data_dir).resolve(), data_dir)
        self._copy_optional(spec.description_path, data_dir / "description.md")
        self._copy_optional(spec.sample_submission_path, data_dir / "sample_submission.csv")
        self._copy_optional(spec.grading_config_path, data_dir / "grading_config.json")
        if spec.validation_command:
            (data_dir / "eval_cmd.txt").write_text(spec.validation_command, encoding="utf-8")
        if not any(code_dir.iterdir()):
            (code_dir / "README.md").write_text(
                "# MLE Code Workspace\n\nAgent-owned repository root.\n",
                encoding="utf-8",
            )

    def _extract_zip(self, archive_path: Path, destination: Path) -> None:
        with ZipFile(archive_path) as zf:
            zf.extractall(destination)

    def _copy_tree(self, source: Path, destination: Path) -> None:
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def _copy_optional(self, source: str | None, destination: Path) -> None:
        if source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source).resolve(), destination)

    def _run_real_loop(self, job: JobRecord, job_paths) -> None:
        profile = default_mle_profile(job.runtime_profile.image_profile)
        image_tag = self.runtime.prepare_image(profile, job.runtime_profile)
        spec = self.runtime.create_session_spec(
            job.id,
            job_paths,
            profile,
            job.runtime_profile,
            layout=WorkspaceLayout.MLE,
            workdir="/home/code",
            env=self._session_env(job),
        )
        session = self.runtime.start_session(spec, image_tag)
        keep_session = False
        try:
            result = self.runtime.exec(
                session,
                "python -m aisci_domain_mle.orchestrator",
                workdir="/home/code",
                check=False,
            )
            if result.stdout:
                append_log(job_paths.logs_dir / "job.log", result.stdout)
            if result.stderr:
                append_log(job_paths.logs_dir / "job.log", result.stderr)
            self._materialize_registry(job_paths, job)
            if result.exit_code != 0:
                if job.runtime_profile.keep_container_on_failure:
                    keep_session = True
                raise DockerRuntimeError(
                    result.combined_output or "MLE orchestrator exited with a non-zero status."
                )
        finally:
            if not keep_session:
                self.runtime.cleanup(session)

    def _session_env(self, job: JobRecord) -> dict[str, str]:
        env = llm_env(job.llm_profile)
        env.update(
            {
                "TIME_LIMIT_SECS": str(self._parse_duration(job.runtime_profile.time_limit)),
                "LOGS_DIR": "/home/logs",
                "COMPETITION_ID": job.id,
                "HARDWARE": self._hardware_label(job),
                "AISCI_CONTEXT_REDUCE_STRATEGY": "summary",
            }
        )
        return env

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

    def _hardware_label(self, job: JobRecord) -> str:
        if job.runtime_profile.gpu_ids:
            return f"gpu:{','.join(job.runtime_profile.gpu_ids)}"
        if job.runtime_profile.gpu_count > 0:
            return f"gpu_count:{job.runtime_profile.gpu_count}"
        return "cpu"

    def _materialize_registry(self, job_paths, job: JobRecord) -> None:
        submission_path = job_paths.workspace_dir / "submission" / "submission.csv"
        registry = CandidateRegistry(
            registry_path=job_paths.workspace_dir / "submission" / "submission_registry.jsonl",
            candidates_dir=job_paths.workspace_dir / "submission" / "candidates",
        )
        if not submission_path.exists():
            return
        registry_text = registry.registry_path.read_text(encoding="utf-8")
        if "system_snapshot" not in registry_text:
            registry.append(
                "system_snapshot",
                objective=job.objective,
                llm_profile=job.llm_profile,
            )
        if "candidate_detail" not in registry_text:
            snapshot = registry.snapshot_submission(
                submission_path,
                reason="final_submission",
                method_summary="Final promoted submission from upstream mle loop.",
                metrics=None,
                eval_protocol="loop_finalize",
            )
            champion_path = snapshot.snapshot
        else:
            candidates = sorted((job_paths.workspace_dir / "submission" / "candidates").glob("*.csv"))
            champion_path = candidates[-1] if candidates else submission_path
        if "champion_selected" not in registry_text:
            registry.select_champion(
                champion_path,
                rationale="Selected latest available candidate after mle loop completion.",
                metrics=None,
                eval_protocol="loop_finalize",
            )
        champion_report = job_paths.artifacts_dir / "champion_report.md"
        champion_report.write_text(
            "# Champion Report\n\n"
            f"- Champion path: `{champion_path}`\n"
            f"- Registry: `{registry.registry_path}`\n",
            encoding="utf-8",
        )

    def _collect_artifacts(self, job_paths) -> list[ArtifactRecord]:
        candidates_dir = job_paths.workspace_dir / "submission" / "candidates"
        known: list[tuple[str, Path, RunPhase, dict[str, object]]] = [
            (
                "analysis_summary",
                job_paths.workspace_dir / "agent" / "analysis" / "summary.md",
                RunPhase.ANALYZE,
                {},
            ),
            (
                "prioritized_tasks",
                job_paths.workspace_dir / "agent" / "prioritized_tasks.md",
                RunPhase.PRIORITIZE,
                {},
            ),
            ("impl_log", job_paths.workspace_dir / "agent" / "impl_log.md", RunPhase.IMPLEMENT, {}),
            ("exp_log", job_paths.workspace_dir / "agent" / "exp_log.md", RunPhase.VALIDATE, {}),
            (
                "submission_registry",
                job_paths.workspace_dir / "submission" / "submission_registry.jsonl",
                RunPhase.FINALIZE,
                {},
            ),
            (
                "submission",
                job_paths.workspace_dir / "submission" / "submission.csv",
                RunPhase.FINALIZE,
                {},
            ),
            ("agent_log", job_paths.logs_dir / "agent.log", RunPhase.FINALIZE, {}),
            (
                "conversation_log",
                job_paths.logs_dir / "conversation.jsonl",
                RunPhase.FINALIZE,
                {},
            ),
            ("summary_json", job_paths.logs_dir / "summary.json", RunPhase.FINALIZE, {}),
            ("champion_report", job_paths.artifacts_dir / "champion_report.md", RunPhase.FINALIZE, {}),
        ]
        artifacts: list[ArtifactRecord] = []
        for artifact_type, path, phase, metadata in known:
            if path.exists():
                artifacts.append(
                    ArtifactRecord(
                        artifact_type=artifact_type,
                        path=str(path),
                        phase=phase,
                        size_bytes=path.stat().st_size,
                        metadata=metadata,
                    )
                )
        if candidates_dir.exists():
            for candidate in sorted(candidates_dir.glob("*.csv")):
                artifacts.append(
                    ArtifactRecord(
                        artifact_type="candidate_snapshot",
                        path=str(candidate),
                        phase=RunPhase.FINALIZE,
                        size_bytes=candidate.stat().st_size,
                        metadata={},
                    )
                )
        return artifacts

    def _maybe_validate(self, job: JobRecord, job_paths) -> ValidationReport:
        profile = default_mle_profile(job.runtime_profile.image_profile)
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
        submission_path = job_paths.workspace_dir / "submission" / "submission.csv"
        if not submission_path.exists():
            return ValidationReport(
                status="failed",
                summary="No submission.csv was staged for validation.",
                details={"reason": "submission_missing"},
                runtime_profile_hash="submission-missing",
                container_image="not-built",
            )
        validation_command = self._validation_command(job.mode_spec, job_paths)
        try:
            image_tag = self.runtime.prepare_image(profile, job.runtime_profile)
            spec = self.runtime.create_session_spec(
                job.id,
                job_paths,
                profile,
                job.runtime_profile,
                layout=WorkspaceLayout.MLE,
                workdir="/home/code",
            )
            return self.runtime.run_validation(
                spec,
                image_tag,
                validation_command,
                workdir="/home/code",
            )
        except DockerRuntimeError as exc:
            return ValidationReport(
                status="failed",
                summary="Final validation failed while preparing the runtime image.",
                details={"reason": str(exc)},
                runtime_profile_hash="runtime-image-failed",
                container_image="image-prepare-failed",
            )

    def _validation_command(self, spec: MLESpec, job_paths) -> str:
        if spec.validation_command:
            return spec.validation_command
        eval_cmd = job_paths.workspace_dir / "data" / "eval_cmd.txt"
        if eval_cmd.exists():
            return eval_cmd.read_text(encoding="utf-8").strip()
        return (
            "python - <<'PY'\n"
            "import csv\n"
            "with open('/home/submission/submission.csv', newline='') as f:\n"
            "    rows = list(csv.reader(f))\n"
            "print(f'rows={len(rows)}')\n"
            "PY"
        )
