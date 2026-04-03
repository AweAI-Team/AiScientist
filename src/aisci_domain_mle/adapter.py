from __future__ import annotations

from dataclasses import dataclass
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
import os
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

from aisci_domain_mle.local_runtime_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

from aisci_agent_runtime.llm_client import LLMConfig, create_llm_client
from aisci_agent_runtime.llm_profiles import backend_env_values, missing_backend_env_vars, resolve_llm_profile
from aisci_core.logging_utils import append_log
from aisci_domain_mle.candidate_registry import CandidateRegistry
from aisci_domain_mle.constants import is_file_as_bus_enabled
from aisci_domain_mle.mlebench_compat import (
    LegacyCompetitionGrader,
    LegacyCompetitionPreparer,
    resolve_competition_source,
)
from aisci_domain_mle.orchestrator_runtime import STUB_ENV_KEYS, runtime_uses_stub_llm
from aisci_domain_mle.orchestrator import EmbeddedMLEEngine
from aisci_domain_mle.orchestrator_runtime import OrchestratorPaths, OrchestratorRuntimeConfig
from aisci_domain_mle.preflight import (
    restore_submission_from_workspace,
    validate_required_public_metadata,
)
from aisci_domain_mle.shared_infra_bridge import (
    default_domain_mle_profile,
    domain_llm_profile_file,
    present_proxy_env_keys,
)
from aisci_runtime_docker.shell_interface import DockerShellInterface

if TYPE_CHECKING:
    from aisci_core.models import ArtifactRecord, JobRecord, MLESpec, ValidationReport
    from aisci_runtime_docker.runtime import DockerRuntimeManager


SENSITIVE_DATA_MARKERS = (
    "answer",
    "gold_submission",
    "solution",
    "test_with_solutions",
    "verification_label",
    "verification_set",
)
SENSITIVE_WORKSPACE_FILENAMES = {
    "eval_cmd.txt",
    "grading_config.json",
}


@dataclass(frozen=True)
class PreparedPublicSource:
    public_dir: Path
    competition_name: str | None = None
    prepared_dir: Path | None = None


@dataclass(frozen=True)
class LegacyValidationTarget:
    competition_name: str
    cache_root: Path
    prepared_dir: Path


class MLEDomainAdapter:
    def __init__(
        self,
        runtime: DockerRuntimeManager,
        *,
        competition_preparer: LegacyCompetitionPreparer | None = None,
        competition_grader: LegacyCompetitionGrader | None = None,
    ):
        self.runtime = runtime
        self._competition_preparer = competition_preparer or LegacyCompetitionPreparer()
        self._competition_grader = competition_grader or LegacyCompetitionGrader()

    def run(self, job: JobRecord) -> dict[str, object]:
        from aisci_core.models import MLESpec, WorkspaceLayout
        from aisci_core.paths import ensure_job_dirs, resolve_job_paths

        assert isinstance(job.mode_spec, MLESpec)
        job_paths = ensure_job_dirs(resolve_job_paths(job.id))
        append_log(job_paths.logs_dir / "job.log", "mle job started")
        self.runtime.ensure_layout(job_paths, WorkspaceLayout.MLE)
        self._stage_inputs(job.mode_spec, job_paths)

        code_dir = job_paths.workspace_dir / "code"
        if not (code_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=code_dir, check=False, capture_output=True)

        llm_profile = self._ensure_runtime_ready(job, job_paths)
        self._write_resolved_llm_config(job_paths, llm_profile)
        self._run_real_loop(job, job_paths, llm_profile)

        artifacts = self._collect_artifacts(job_paths)
        validation = self._maybe_validate(job, job_paths)
        return {"artifacts": artifacts, "validation_report": validation}

    def _resolve_llm_profile(self, job: JobRecord):
        return resolve_llm_profile(
            job.llm_profile,
            default_for="mle",
            profile_file=domain_llm_profile_file(),
        )

    def _ensure_runtime_ready(self, job: JobRecord, job_paths):
        from aisci_runtime_docker.runtime import DockerRuntimeError

        if not self.runtime.can_use_docker():
            message = "MLE mode requires a reachable Docker daemon. No local fallback loop is available."
            append_log(job_paths.logs_dir / "job.log", message)
            raise DockerRuntimeError(message)
        profile = self._resolve_llm_profile(job)
        if runtime_uses_stub_llm():
            append_log(
                job_paths.logs_dir / "job.log",
                f"resolved llm profile={profile.name} backend={profile.backend_name} "
                f"provider={profile.provider} model={profile.model} api={profile.api_mode} "
                "(stub mode; backend credential check skipped)",
            )
            return profile
        missing = missing_backend_env_vars(profile)
        if missing:
            message = (
                f"MLE mode requires backend credentials for profile {profile.name}: "
                f"{', '.join(missing)}. No local fallback loop is available."
            )
            append_log(job_paths.logs_dir / "job.log", message)
            raise RuntimeError(message)
        append_log(
            job_paths.logs_dir / "job.log",
            f"resolved llm profile={profile.name} backend={profile.backend_name} "
            f"provider={profile.provider} model={profile.model} api={profile.api_mode}",
        )
        return profile

    def _write_resolved_llm_config(self, job_paths, profile) -> None:
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
                "web_search_requested": False,
                "web_search_enabled": False,
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

    def _write_session_info(
        self,
        job_paths,
        session,
        llm_profile: str,
        session_env: dict[str, str],
        *,
        kept_on_failure: bool = False,
    ) -> None:
        inspect_summary = self.runtime.inspect_session(session)
        payload = {
            "container_name": session.container_name,
            "image_ref": session.image_tag,
            "runtime_image_profile": session.profile.name,
            "pull_policy": (
                session.runtime_profile.pull_policy.value
                if session.runtime_profile.pull_policy
                else session.profile.pull_policy.value
            ),
            "workdir": session.workdir,
            "run_as_user": session.run_as_user,
            "started_at": session.started_at.isoformat(),
            "llm_profile": llm_profile,
            "kept_on_failure": kept_on_failure,
            "labels": {key: value for key, value in session.labels},
            "forwarded_env": {key: {"present": True} for key in sorted(session_env)},
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

    def _stage_inputs(self, spec: MLESpec, job_paths) -> None:
        data_dir = job_paths.workspace_dir / "data"
        code_dir = job_paths.workspace_dir / "code"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)

        if spec.code_repo_zip:
            bundle = Path(spec.code_repo_zip).resolve()
            shutil.copy2(bundle, job_paths.input_dir / "code_repo.zip")
            self._extract_zip(bundle, code_dir)

        if spec.grading_config_path:
            raise ValueError(
                "grading_config_path is not supported in the live MLE adapter because it can "
                "expose private grading data to the solving workspace."
            )

        with self._resolve_public_data_source(spec) as source:
            self._assert_public_data_safe(source.public_dir)
            self._copy_tree(source.public_dir, data_dir)
            self._stage_public_metadata(
                spec,
                destination=data_dir,
                competition_name=source.competition_name,
                prepared_dir=source.prepared_dir,
            )
        validate_required_public_metadata(data_dir)
        if not any(code_dir.iterdir()):
            (code_dir / "README.md").write_text(
                "# MLE Code Workspace\n\nAgent-owned repository root.\n",
                encoding="utf-8",
            )

    @contextmanager
    def _resolve_public_data_source(self, spec: MLESpec):
        self._ensure_single_competition_data_source(spec)

        bundle_path = getattr(spec, "workspace_bundle_zip", None) or getattr(
            spec, "competition_bundle_zip", None
        )
        if bundle_path:
            resolved_bundle_path = Path(bundle_path).resolve()
            competition_name = self._competition_name_from_bundle(resolved_bundle_path)
            with self._prepare_local_zip_source(
                resolved_bundle_path,
                competition_name=competition_name,
            ) as source:
                yield source
            return

        competition_name = getattr(spec, "competition_name", None)
        competition_zip_path = getattr(spec, "competition_zip_path", None)
        if competition_name or competition_zip_path:
            resolved_inputs = resolve_competition_source(
                competition_name=competition_name,
                competition_zip_path=competition_zip_path,
                cache_root=self._mlebench_cache_root(spec),
                allow_download=False,
            )
            if resolved_inputs.competition_zip_path:
                zip_path = Path(resolved_inputs.competition_zip_path).resolve()
                prepared_name = resolved_inputs.competition_name or self._competition_name_from_bundle(zip_path)
                with self._prepare_local_zip_source(
                    zip_path,
                    competition_name=prepared_name,
                ) as source:
                    yield source
                return

            if resolved_inputs.cache_prepared_exists and resolved_inputs.cache_prepared_dir:
                prepared_dir = Path(resolved_inputs.cache_prepared_dir).resolve()
                yield PreparedPublicSource(
                    public_dir=prepared_dir / "public",
                    competition_name=resolved_inputs.competition_name,
                    prepared_dir=prepared_dir,
                )
                return

            raise ValueError(self._competition_prepare_message(resolved_inputs))

        data_dir = getattr(spec, "data_dir", None)
        if data_dir:
            yield self._resolve_safe_data_dir(Path(data_dir).resolve())
            return

        raise ValueError("mle job requires a competition data source")

    def _ensure_single_competition_data_source(self, spec: MLESpec) -> None:
        configured_sources: list[str] = []
        if getattr(spec, "competition_zip_path", None):
            configured_sources.append("competition_zip_path")
        elif getattr(spec, "competition_name", None):
            configured_sources.append("competition_name")
        configured_sources.extend(
            name
            for name, raw in (
                ("workspace_bundle_zip", getattr(spec, "workspace_bundle_zip", None)),
                ("competition_bundle_zip", getattr(spec, "competition_bundle_zip", None)),
                ("data_dir", getattr(spec, "data_dir", None)),
            )
            if raw
        )
        if len(configured_sources) != 1:
            joined = ", ".join(configured_sources) if configured_sources else "none"
            raise ValueError(
                "live MLE adapter requires exactly one competition data source; "
                f"got {joined}"
            )

    @contextmanager
    def _prepare_local_zip_source(
        self,
        bundle_path: Path,
        *,
        competition_name: str,
    ):
        with tempfile.TemporaryDirectory(
            prefix=f"aisci-mle-{competition_name}-",
            dir="/tmp",
        ) as temp_dir:
            staging_root = Path(temp_dir).resolve()
            raw_dir = staging_root / "raw"
            prepared_dir = staging_root / "prepared"
            public_dir = prepared_dir / "public"
            private_dir = prepared_dir / "private"
            raw_dir.mkdir(parents=True, exist_ok=True)
            public_dir.mkdir(parents=True, exist_ok=True)
            private_dir.mkdir(parents=True, exist_ok=True)

            self._extract_zip(bundle_path, raw_dir)
            effective_raw_dir = self._unwrap_single_directory(raw_dir)
            self._competition_preparer.prepare_local_dataset(
                competition_name,
                raw_dir=effective_raw_dir,
                public_dir=public_dir,
                private_dir=private_dir,
            )
            yield PreparedPublicSource(
                public_dir=public_dir,
                competition_name=competition_name,
                prepared_dir=prepared_dir,
            )

    @contextmanager
    def _prepare_local_validation_cache(
        self,
        bundle_path: Path,
        *,
        competition_name: str,
    ):
        with tempfile.TemporaryDirectory(
            prefix=f"aisci-mle-validate-{competition_name}-",
            dir="/tmp",
        ) as temp_dir:
            cache_root = Path(temp_dir).resolve()
            competition_root = cache_root / competition_name
            raw_dir = competition_root / "raw"
            prepared_dir = competition_root / "prepared"
            public_dir = prepared_dir / "public"
            private_dir = prepared_dir / "private"
            raw_dir.mkdir(parents=True, exist_ok=True)
            public_dir.mkdir(parents=True, exist_ok=True)
            private_dir.mkdir(parents=True, exist_ok=True)

            self._extract_zip(bundle_path, raw_dir)
            effective_raw_dir = self._unwrap_single_directory(raw_dir)
            self._competition_preparer.prepare_local_dataset(
                competition_name,
                raw_dir=effective_raw_dir,
                public_dir=public_dir,
                private_dir=private_dir,
            )
            yield LegacyValidationTarget(
                competition_name=competition_name,
                cache_root=cache_root,
                prepared_dir=prepared_dir,
            )

    def _mlebench_cache_root(self, spec: MLESpec) -> Path:
        configured = getattr(spec, "mlebench_data_dir", None)
        if configured:
            return Path(configured).expanduser().resolve()
        return (Path.home() / ".cache" / "mle-bench" / "data").resolve()

    def _competition_prepare_message(self, resolved_inputs) -> str:
        command = " ".join(resolved_inputs.legacy_prepare_plan.command) if resolved_inputs.legacy_prepare_plan else ""
        detail = (
            f" Run `proxy-on` first, then `{command}`."
            if command
            else " Run the built-in MLE prepare flow first."
        )
        return (
            "Prepared MLE-Bench cache is missing for competition "
            f"{resolved_inputs.competition_name!r}.{detail}"
        )

    def _resolve_safe_data_dir(self, source: Path) -> PreparedPublicSource:
        if source.name == "public" and source.is_dir():
            prepared_dir = source.parent if source.parent.name == "prepared" else None
            competition_name = (
                source.parent.parent.name if prepared_dir is not None else None
            ) or None
            return PreparedPublicSource(
                public_dir=source,
                competition_name=competition_name,
                prepared_dir=prepared_dir,
            )

        if source.name == "prepared" and (source / "public").is_dir():
            competition_name = source.parent.name or None
            return PreparedPublicSource(
                public_dir=source / "public",
                competition_name=competition_name,
                prepared_dir=source,
            )

        if (source / "prepared" / "public").is_dir():
            return PreparedPublicSource(
                public_dir=source / "prepared" / "public",
                competition_name=source.name or None,
                prepared_dir=source / "prepared",
            )

        raise ValueError(
            "data_dir must point to a public competition directory, a prepared directory, "
            "or a competition root containing prepared/public."
        )

    def _stage_public_metadata(
        self,
        spec: MLESpec,
        *,
        destination: Path,
        competition_name: str | None,
        prepared_dir: Path | None,
    ) -> None:
        description_source = (
            self._validated_public_metadata_override(Path(spec.description_path).resolve(), "description_path")
            if spec.description_path
            else None
        )
        sample_submission_source = (
            self._validated_public_metadata_override(
                Path(spec.sample_submission_path).resolve(),
                "sample_submission_path",
            )
            if spec.sample_submission_path
            else None
        )

        if prepared_dir is not None and competition_name:
            try:
                legacy_description, legacy_sample_submission = (
                    self._competition_preparer.resolve_public_metadata_paths(
                        competition_name,
                        prepared_dir=prepared_dir,
                    )
                )
            except ValueError:
                legacy_description = None
                legacy_sample_submission = None
            description_source = description_source or legacy_description
            sample_submission_source = sample_submission_source or legacy_sample_submission

        self._copy_if_present(description_source, destination / "description.md")
        self._copy_if_present(sample_submission_source, destination / "sample_submission.csv")

    def _validated_public_metadata_override(self, path: Path, label: str) -> Path:
        lowered_parts = {part.lower() for part in path.parts}
        lowered_name = path.name.lower()
        if any(part.startswith("private") for part in lowered_parts):
            raise ValueError(f"{label} must not point to private competition data.")
        if lowered_name in SENSITIVE_WORKSPACE_FILENAMES:
            raise ValueError(f"{label} must not point to validation or grading artifacts.")
        if any(marker in lowered_name for marker in SENSITIVE_DATA_MARKERS):
            raise ValueError(f"{label} must not point to private competition data.")
        return path

    def _copy_if_present(self, source: Path | None, destination: Path) -> None:
        if source is None:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.resolve(), destination)

    def _assert_public_data_safe(self, root: Path) -> None:
        root = root.resolve()
        for path in root.rglob("*"):
            lowered_parts = {part.lower() for part in path.parts}
            lowered_name = path.name.lower()
            if any(part.startswith("private") for part in lowered_parts):
                raise ValueError("public competition staging contains private paths.")
            if lowered_name in SENSITIVE_WORKSPACE_FILENAMES:
                raise ValueError("public competition staging contains validation or grading artifacts.")
            if any(marker in lowered_name for marker in SENSITIVE_DATA_MARKERS):
                raise ValueError("public competition staging contains files that look private.")

    def _competition_name_from_bundle(self, bundle_path: Path) -> str:
        name = bundle_path.name
        if name.lower().endswith(".zip"):
            name = name[:-4]
        cleaned = name.strip()
        if not cleaned:
            raise ValueError(f"could not infer competition name from bundle path {bundle_path}")
        return cleaned

    def _unwrap_single_directory(self, root: Path) -> Path:
        entries = [entry for entry in root.iterdir() if entry.name != "__MACOSX"]
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
        return root

    def _extract_zip(self, archive_path: Path, destination: Path) -> None:
        with ZipFile(archive_path) as zf:
            destination = destination.resolve()
            for member in zf.infolist():
                member_path = (destination / member.filename).resolve()
                if member_path != destination and destination not in member_path.parents:
                    raise ValueError(f"zip archive contains unsafe path {member.filename!r}")
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

    def _run_real_loop(self, job: JobRecord, job_paths, llm_profile) -> None:
        from aisci_core.models import WorkspaceLayout

        profile = default_domain_mle_profile()
        image_tag = self.runtime.prepare_image(profile, job.runtime_profile)
        session_env = self._session_env(job)
        spec = self.runtime.create_session_spec(
            job.id,
            job_paths,
            profile,
            job.runtime_profile,
            layout=WorkspaceLayout.MLE,
            workdir="/home/code",
            env=session_env,
        )
        session = self.runtime.start_session(spec, image_tag)
        self._write_session_info(
            job_paths,
            session,
            llm_profile=llm_profile.name,
            session_env=session_env,
        )
        keep_session = False
        try:
            shell = DockerShellInterface(self.runtime, session, working_dir="/home/code")
            engine = EmbeddedMLEEngine(
                config=self._orchestrator_runtime(job, llm_profile),
                shell=shell,
                llm=self._build_llm_client(llm_profile),
            )
            append_log(job_paths.logs_dir / "job.log", "starting host-side mle engine")
            summary = engine.run()
            append_log(job_paths.logs_dir / "job.log", summary)
        except Exception:
            if job.runtime_profile.keep_container_on_failure:
                keep_session = True
                append_log(
                    job_paths.logs_dir / "job.log",
                    f"mle sandbox preserved for debugging: {session.container_name}",
                )
                self._write_session_info(
                    job_paths,
                    session,
                    llm_profile=llm_profile.name,
                    session_env=session_env,
                    kept_on_failure=True,
                )
            raise
        finally:
            rescued_submission = restore_submission_from_workspace(job_paths.workspace_dir)
            if rescued_submission is not None:
                append_log(
                    job_paths.logs_dir / "job.log",
                    f"submission.csv is present at {rescued_submission}",
                )
            else:
                append_log(
                    job_paths.logs_dir / "job.log",
                    "WARNING: submission.csv not found under workspace/submission or known code fallback paths",
                )
            try:
                self._materialize_registry(job_paths, job)
            except Exception as exc:  # noqa: BLE001
                append_log(
                    job_paths.logs_dir / "job.log",
                    f"WARNING: failed to materialize submission registry artifacts: {exc}",
                )
            if not keep_session:
                self.runtime.cleanup(session)

    def _session_env(self, job: JobRecord) -> dict[str, str]:
        env = {
            "AISCI_JOB_ID": job.id,
            "AISCI_OBJECTIVE": job.objective,
            "LOGS_DIR": "/home/logs",
        }
        for key in present_proxy_env_keys():
            value = os.environ.get(key)
            if value:
                env[key] = value
        for key in STUB_ENV_KEYS:
            if key == "LOGS_DIR":
                continue
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _build_llm_client(self, profile):
        backend_values = backend_env_values(profile)
        if profile.name == "glm-5":
            backend_values.setdefault("api_version", "2024-02-01")
        elif profile.name == "gemini-3-flash":
            backend_values.setdefault("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
        return create_llm_client(
            LLMConfig(
                provider=profile.provider,
                model=profile.model,
                api_mode=profile.api_mode,
                max_tokens=profile.max_tokens,
                reasoning_effort=profile.reasoning_effort,
                reasoning_summary=profile.reasoning_summary,
                web_search=bool(profile.web_search and profile.api_mode == "responses"),
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

    def _orchestrator_runtime(self, job: JobRecord, profile) -> OrchestratorRuntimeConfig:
        return OrchestratorRuntimeConfig(
            time_limit=self._parse_duration(job.runtime_profile.time_limit),
            max_steps=int(os.environ.get("AISCI_MAX_STEPS", "500")),
            reminder_freq=int(os.environ.get("AISCI_REMINDER_FREQ", "5")),
            model=profile.model,
            hardware=self._hardware_label(job),
            api_mode=profile.api_mode,
            context_reduce_strategy=(os.environ.get("AISCI_CONTEXT_REDUCE_STRATEGY", "summary") or "summary").strip().lower(),
            summary_segment_ratio=float(os.environ.get("AISCI_SUMMARY_SEGMENT_RATIO", "0.3")),
            summary_min_turns=int(os.environ.get("AISCI_SUMMARY_MIN_TURNS_TO_SUMMARIZE", "4")),
            summary_segment_max_chars=int(os.environ.get("AISCI_SUMMARY_SEGMENT_MAX_CHARS", "25000")),
            summary_incremental=(os.environ.get("AISCI_SUMMARY_INCREMENTAL", "true") or "true").strip().lower() in ("true", "1", "yes"),
            file_as_bus=is_file_as_bus_enabled(),
            paths=OrchestratorPaths(
                home_root="/home",
                data_dir="/home/data",
                code_dir="/home/code",
                submission_dir="/home/submission",
                agent_dir="/home/agent",
                logs_dir="/home/logs",
            ),
            validation_command=(job.mode_spec.validation_command or "").strip() or None,
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
        registry_text = (
            registry.registry_path.read_text(encoding="utf-8")
            if registry.registry_path.exists()
            else ""
        )
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
        from aisci_core.models import ArtifactRecord, RunPhase

        candidates_dir = job_paths.workspace_dir / "submission" / "candidates"
        known: list[tuple[str, Path, RunPhase, dict[str, object]]] = [
            (
                "resolved_llm_config",
                job_paths.state_dir / "resolved_llm_config.json",
                RunPhase.INGEST,
                {},
            ),
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
            ("sandbox_session", job_paths.state_dir / "sandbox_session.json", RunPhase.FINALIZE, {}),
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

    @contextmanager
    def _resolve_legacy_validation_target(self, spec: MLESpec):
        self._ensure_single_competition_data_source(spec)

        bundle_path = getattr(spec, "workspace_bundle_zip", None) or getattr(
            spec, "competition_bundle_zip", None
        )
        if bundle_path:
            resolved_bundle_path = Path(bundle_path).resolve()
            competition_name = self._competition_name_from_bundle(resolved_bundle_path)
            with self._prepare_local_validation_cache(
                resolved_bundle_path,
                competition_name=competition_name,
            ) as target:
                yield target
            return

        competition_name = getattr(spec, "competition_name", None)
        competition_zip_path = getattr(spec, "competition_zip_path", None)
        if competition_name or competition_zip_path:
            resolved_inputs = resolve_competition_source(
                competition_name=competition_name,
                competition_zip_path=competition_zip_path,
                cache_root=self._mlebench_cache_root(spec),
                allow_download=False,
            )
            if resolved_inputs.cache_prepared_exists and resolved_inputs.cache_prepared_dir:
                prepared_dir = Path(resolved_inputs.cache_prepared_dir).resolve()
                if (prepared_dir / "private").is_dir():
                    yield LegacyValidationTarget(
                        competition_name=resolved_inputs.competition_name or prepared_dir.parent.name,
                        cache_root=Path(resolved_inputs.cache_root).resolve(),
                        prepared_dir=prepared_dir,
                    )
                    return
            if resolved_inputs.competition_zip_path:
                zip_path = Path(resolved_inputs.competition_zip_path).resolve()
                prepared_name = resolved_inputs.competition_name or self._competition_name_from_bundle(zip_path)
                with self._prepare_local_validation_cache(
                    zip_path,
                    competition_name=prepared_name,
                ) as target:
                    yield target
                return
            yield None
            return

        data_dir = getattr(spec, "data_dir", None)
        if data_dir:
            source = self._resolve_safe_data_dir(Path(data_dir).resolve())
            if source.prepared_dir is None or not source.competition_name:
                yield None
                return
            if not (source.prepared_dir / "private").is_dir():
                yield None
                return
            yield LegacyValidationTarget(
                competition_name=source.competition_name,
                cache_root=source.prepared_dir.parent.parent.resolve(),
                prepared_dir=source.prepared_dir.resolve(),
            )
            return

        yield None

    def _run_legacy_grade_validation(self, submission_path: Path, target: LegacyValidationTarget) -> ValidationReport:
        from aisci_core.models import ValidationReport

        grade_report = self._competition_grader.grade_submission(
            submission_path,
            competition_name=target.competition_name,
            cache_root=target.cache_root,
        )
        status = "passed" if grade_report.get("valid_submission") else "failed"
        if status == "passed":
            summary = (
                f"Legacy MLE-Bench grading completed for {target.competition_name}: "
                f"score={grade_report.get('score')}"
            )
        else:
            reason = grade_report.get("error") or "submission could not be graded"
            summary = (
                f"Legacy MLE-Bench grading failed for {target.competition_name}: "
                f"{reason}"
            )
        return ValidationReport(
            status=status,
            summary=summary,
            details={
                "grader": "legacy_mlebench_host",
                **grade_report,
            },
            runtime_profile_hash="legacy-grade",
            container_image="host:mlebench",
        )

    def _maybe_validate(self, job: JobRecord, job_paths) -> ValidationReport:
        from aisci_core.models import ValidationReport, WorkspaceLayout
        from aisci_runtime_docker.runtime import DockerRuntimeError

        profile = default_domain_mle_profile()
        if not job.runtime_profile.run_final_validation:
            return ValidationReport(
                status="skipped",
                summary="Final validation disabled for this job.",
                runtime_profile_hash="disabled",
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
        if not job.mode_spec.validation_command:
            with self._resolve_legacy_validation_target(job.mode_spec) as target:
                if target is not None:
                    return self._run_legacy_grade_validation(submission_path, target)
        if not self.runtime.can_use_docker():
            return ValidationReport(
                status="failed",
                summary="Docker is not available on this machine.",
                details={"reason": "docker_not_found"},
                runtime_profile_hash="docker-missing",
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
