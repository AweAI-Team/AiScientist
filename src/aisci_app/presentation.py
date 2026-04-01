from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisci_agent_runtime.llm_profiles import (
    missing_backend_env_vars,
    resolve_llm_profile,
    resolved_profile_path,
)
from aisci_core.models import JobRecord, JobSpec, JobType, MLESpec, PaperSpec, PullPolicy, RuntimeProfile, WorkspaceLayout
from aisci_core.paths import repo_root, resolve_job_paths
from aisci_runtime_docker.profiles import (
    resolve_image_profile,
    resolved_image_profile_path,
)


@dataclass(frozen=True)
class PaperDoctorCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class PaperLogTarget:
    label: str
    path: str
    exists: bool
    kind: str


@dataclass(frozen=True)
class PaperArtifactHint:
    label: str
    path: str
    exists: bool
    purpose: str


def _job_root(job: JobRecord) -> Path:
    return resolve_job_paths(job.id).root


def _paper_paths(job: JobRecord) -> dict[str, Path]:
    job_paths = resolve_job_paths(job.id)
    return {
        "root": job_paths.root,
        "workspace": job_paths.workspace_dir,
        "logs": job_paths.logs_dir,
        "artifacts": job_paths.artifacts_dir,
        "analysis": job_paths.workspace_dir / "agent" / "paper_analysis",
        "submission": job_paths.workspace_dir / "submission",
    }


def _resolved_paper_capabilities(job: JobRecord) -> dict[str, Any]:
    if job.job_type != JobType.PAPER:
        return {}
    path = resolve_job_paths(job.id).state_dir / "capabilities.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def paper_capability_flags(job: JobRecord) -> dict[str, str]:
    if job.job_type != JobType.PAPER:
        return {}
    assert isinstance(job.mode_spec, PaperSpec)
    runtime = job.runtime_profile
    resolved = _resolved_paper_capabilities(job)
    online = resolved.get("online_research")
    linter = resolved.get("linter")
    return {
        "online_research": (
            "available"
            if isinstance(online, dict) and online.get("available")
            else ("requested" if job.mode_spec.enable_online_research else "disabled")
        ),
        "linter": "available" if isinstance(linter, dict) and linter.get("available", False) else "enabled",
        "final_self_check": "enabled" if runtime.run_final_validation else "disabled",
        "validation_strategy": runtime.validation_strategy.value,
        "workspace_layout": runtime.workspace_layout.value if runtime.workspace_layout else "paper",
        "runtime_image": runtime.image or f"profile:{runtime.image_profile or 'default'}",
        "pull_policy": runtime.pull_policy.value if runtime.pull_policy else "profile-default",
    }


def paper_log_targets(job: JobRecord) -> list[PaperLogTarget]:
    if job.job_type != JobType.PAPER:
        return []
    paths = _paper_paths(job)
    logs_dir = paths["logs"]
    subagent_dir = logs_dir / "subagent_logs"
    candidates = [
        ("main job log", logs_dir / "job.log", "main"),
        ("worker log", logs_dir / "worker.log", "main"),
        ("conversation log", logs_dir / "conversation.jsonl", "conversation"),
        ("agent log", logs_dir / "agent.log", "agent"),
        ("summary log", logs_dir / "context_summary_requests.jsonl", "summary"),
        ("subagent logs", subagent_dir, "subagent"),
        ("session state", logs_dir / "paper_session_state.json", "state"),
        ("sandbox session", resolve_job_paths(job.id).state_dir / "sandbox_session.json", "sandbox"),
        ("validation report", paths["artifacts"] / "validation_report.json", "validation"),
        ("self-check report", paths["workspace"] / "agent" / "final_self_check.md", "validation"),
    ]
    if subagent_dir.exists():
        for session_dir in sorted(path for path in subagent_dir.iterdir() if path.is_dir())[:20]:
            candidates.append((f"session: {session_dir.name}", session_dir, "subagent_session"))
    return [PaperLogTarget(label, str(path), path.exists(), kind) for label, path, kind in candidates]


def paper_artifact_hints(job: JobRecord) -> list[PaperArtifactHint]:
    if job.job_type != JobType.PAPER:
        return []
    paths = _paper_paths(job)
    job_paths = resolve_job_paths(job.id)
    analysis = paths["analysis"]
    submission = paths["submission"]
    candidates = [
        ("paper summary", analysis / "summary.md", "Read this first to understand the paper."),
        ("paper structure", analysis / "structure.md", "Section map and line numbers."),
        ("paper algorithm", analysis / "algorithm.md", "Core method and implementation details."),
        ("paper experiments", analysis / "experiments.md", "Dataset, metrics, and experiment plan."),
        ("paper baseline", analysis / "baseline.md", "Baseline methods and comparison set."),
        ("prioritized tasks", paths["workspace"] / "agent" / "prioritized_tasks.md", "Ranked implementation plan."),
        ("plan", paths["workspace"] / "agent" / "plan.md", "Current planning note for auxiliary planning work."),
        ("implementation log", paths["workspace"] / "agent" / "impl_log.md", "Changelog for code changes."),
        ("experiment log", paths["workspace"] / "agent" / "exp_log.md", "Experiment history and results."),
        ("capability report", job_paths.state_dir / "capabilities.json", "Resolved online research and validation capabilities."),
        ("resolved llm config", job_paths.state_dir / "resolved_llm_config.json", "Resolved backend, model, and token settings selected on the host."),
        ("main prompt snapshot", job_paths.state_dir / "paper_main_prompt.md", "Final rendered main-agent prompt used for this run."),
        ("sandbox session", job_paths.state_dir / "sandbox_session.json", "Persistent Docker sandbox metadata for this run."),
        ("self-check report", paths["workspace"] / "agent" / "final_self_check.md", "Latest clean reproducibility report."),
        ("self-check report json", paths["workspace"] / "agent" / "final_self_check.json", "Structured self-check result for automation and UI."),
        ("reproduce script", submission / "reproduce.sh", "Entry point for local and final self-check."),
    ]
    return [PaperArtifactHint(label, str(path), path.exists(), purpose) for label, path, purpose in candidates]


def build_job_spec_clone(
    job: JobRecord,
    *,
    objective_suffix: str = "",
    run_final_validation: bool | None = None,
) -> JobSpec:
    runtime = job.runtime_profile.model_copy(deep=True)
    if run_final_validation is not None:
        runtime.run_final_validation = run_final_validation
    mode_spec = job.mode_spec.model_copy(deep=True)
    if job.job_type == JobType.PAPER:
        assert isinstance(mode_spec, PaperSpec)
    else:
        assert isinstance(mode_spec, MLESpec)
    return JobSpec(
        job_type=job.job_type,
        objective=(job.objective + objective_suffix).strip(),
        llm_profile=job.llm_profile,
        runtime_profile=runtime,
        mode_spec=mode_spec,
    )


def build_paper_job_spec(
    *,
    pdf_path: str | None,
    paper_bundle_zip: str | None,
    paper_md_path: str | None,
    llm_profile: str,
    gpus: int,
    time_limit: str,
    inputs_zip: str | None,
    rubric_path: str | None,
    blacklist_path: str | None,
    addendum_path: str | None,
    seed_repo_zip: str | None,
    supporting_materials: list[str],
    image: str | None,
    pull_policy: PullPolicy | None,
    run_final_validation: bool,
    gpu_ids: list[str] | None = None,
    enable_online_research: bool = True,
    objective: str = "paper reproduction job",
) -> JobSpec:
    runtime = RuntimeProfile(
        gpu_count=0 if gpu_ids else gpus,
        gpu_ids=list(gpu_ids or []),
        time_limit=time_limit,
        image=image,
        pull_policy=pull_policy,
        run_final_validation=run_final_validation,
        workspace_layout=WorkspaceLayout.PAPER,
    )
    return JobSpec(
        job_type=JobType.PAPER,
        objective=objective,
        llm_profile=llm_profile,
        runtime_profile=runtime,
        mode_spec=PaperSpec(
            pdf_path=pdf_path,
            paper_bundle_zip=paper_bundle_zip,
            paper_md_path=paper_md_path,
            context_bundle_zip=inputs_zip,
            rubric_path=rubric_path,
            blacklist_path=blacklist_path,
            addendum_path=addendum_path,
            supporting_materials=supporting_materials,
            submission_seed_repo_zip=seed_repo_zip,
            enable_online_research=enable_online_research,
        ),
    )


def build_mle_job_spec(
    *,
    workspace_zip: str | None,
    competition_bundle_zip: str | None,
    data_dir: str | None,
    code_repo_zip: str | None,
    description_path: str | None,
    sample_submission_path: str | None,
    validation_command: str | None,
    grading_config_path: str | None,
    metric_direction: str | None,
    llm_profile: str,
    gpus: int,
    time_limit: str,
    image: str | None,
    pull_policy: PullPolicy | None,
    run_final_validation: bool,
    gpu_ids: list[str] | None = None,
    objective: str = "mle optimization job",
) -> JobSpec:
    runtime = RuntimeProfile(
        gpu_count=0 if gpu_ids else gpus,
        gpu_ids=list(gpu_ids or []),
        time_limit=time_limit,
        image=image,
        pull_policy=pull_policy,
        run_final_validation=run_final_validation,
        workspace_layout=WorkspaceLayout.MLE,
    )
    return JobSpec(
        job_type=JobType.MLE,
        objective=objective,
        llm_profile=llm_profile,
        runtime_profile=runtime,
        mode_spec=MLESpec(
            workspace_bundle_zip=workspace_zip,
            competition_bundle_zip=competition_bundle_zip,
            data_dir=data_dir,
            code_repo_zip=code_repo_zip,
            description_path=description_path,
            sample_submission_path=sample_submission_path,
            validation_command=validation_command,
            grading_config_path=grading_config_path,
            metric_direction=metric_direction,
        ),
    )


def list_text_tree(root: Path, *, max_depth: int = 4, max_entries: int = 250) -> list[dict[str, Any]]:
    root = root.resolve()
    nodes: list[dict[str, Any]] = []
    count = 0
    if not root.exists():
        return nodes

    def walk(path: Path, depth: int) -> None:
        nonlocal count
        if count >= max_entries or depth > max_depth:
            return
        rel = "." if path == root else str(path.relative_to(root))
        nodes.append(
            {
                "name": path.name or root.name,
                "relative_path": rel,
                "depth": depth,
                "is_dir": path.is_dir(),
                "size": path.stat().st_size if path.is_file() else None,
            }
        )
        count += 1
        if path.is_dir():
            children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            for child in children:
                walk(child, depth + 1)

    walk(root, 0)
    return nodes


def read_text_preview(path: Path, *, max_chars: int = 12000) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        return {"path": str(path), "exists": False, "kind": "missing", "content": ""}
    if path.is_dir():
        return {"path": str(path), "exists": True, "kind": "directory", "content": ""}
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return {
            "path": str(path),
            "exists": True,
            "kind": "pdf",
            "content": "Binary PDF preview is not embedded. Download the file to inspect it locally.",
        }
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return {
            "path": str(path),
            "exists": True,
            "kind": "image",
            "content": "Binary image preview is not embedded.",
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[truncated]..."
    return {
        "path": str(path),
        "exists": True,
        "kind": "text",
        "content": text,
    }


def paper_doctor_report() -> list[PaperDoctorCheck]:
    checks = [
        PaperDoctorCheck("repo", "ok" if repo_root().exists() else "fail", f"root={repo_root()}"),
        PaperDoctorCheck(
            "app package",
            "ok" if (repo_root() / "src" / "aisci_app").exists() else "fail",
            "agent console code is present",
        ),
        PaperDoctorCheck(
            "paper workspace",
            "ok" if (repo_root() / "src" / "aisci_domain_paper").exists() else "warn",
            "paper execution path is available",
        ),
    ]
    docker = shutil.which("docker")
    if docker:
        try:
            result = subprocess.run(
                [docker, "info"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            status = "ok" if result.returncode == 0 else "fail"
            detail = (
                "docker daemon reachable"
                if result.returncode == 0
                else ((result.stderr or result.stdout or "docker info failed") + "; paper jobs will not start without Docker")
            )
        except Exception as exc:
            status = "fail"
            detail = f"{exc}; paper jobs will not start without Docker"
    else:
        status = "fail"
        detail = "docker binary not found; paper jobs will not start without Docker"
    checks.append(PaperDoctorCheck("docker", status, detail))
    image_profile_path = resolved_image_profile_path()
    if image_profile_path.exists():
        checks.append(
            PaperDoctorCheck(
                "runtime_image_config",
                "ok",
                f"Using {image_profile_path}",
            )
        )
    else:
        checks.append(
            PaperDoctorCheck(
                "runtime_image_config",
                "fail",
                f"Missing runtime image config file: {image_profile_path}",
            )
        )
    try:
        runtime_image = resolve_image_profile(None, default_for="paper")
    except Exception as exc:  # noqa: BLE001
        checks.append(PaperDoctorCheck("runtime_image", "fail", str(exc)))
    else:
        checks.append(
            PaperDoctorCheck(
                "runtime_image",
                "ok",
                f"profile={runtime_image.name} image={runtime_image.image} pull_policy={runtime_image.pull_policy.value}",
            )
        )
    requested_profile = os.environ.get("AISCI_PAPER_DOCTOR_PROFILE")
    profile_path = resolved_profile_path()
    if profile_path.exists():
        checks.append(
            PaperDoctorCheck(
                "llm_config",
                "ok",
                f"Using {profile_path}",
            )
        )
    else:
        checks.append(
            PaperDoctorCheck(
                "llm_config",
                "fail",
                f"Missing LLM config file: {profile_path}",
            )
        )
    try:
        profile = resolve_llm_profile(requested_profile, default_for="paper")
    except Exception as exc:  # noqa: BLE001
        checks.append(
            PaperDoctorCheck(
                "llm_profile",
                "fail",
                str(exc),
            )
        )
        checks.append(
            PaperDoctorCheck(
                "api_key",
                "fail",
                "LLM profile could not be resolved, so backend credentials could not be checked.",
            )
        )
        profile = None
    if profile is not None:
        profile_name = requested_profile or profile.name
        checks.append(
            PaperDoctorCheck(
                "llm_profile",
                "ok",
                f"name={profile_name} backend={profile.backend_name} provider={profile.provider} model={profile.model} api_mode={profile.api_mode}",
            )
        )
        missing_env = missing_backend_env_vars(profile)
        checks.append(
            PaperDoctorCheck(
                "api_key",
                "ok" if not missing_env else "fail",
                (
                    "Backend credentials detected"
                    if not missing_env
                    else f"Missing backend env vars for profile {profile.name}: {', '.join(missing_env)}"
                ),
            )
        )
    web_search_enabled = os.environ.get("AISCI_WEB_SEARCH", "").strip().lower() in {"1", "true", "yes", "on"}
    checks.append(
        PaperDoctorCheck(
            "online_research",
            "ok" if web_search_enabled else "warn",
            "AISCI_WEB_SEARCH is enabled" if web_search_enabled else "AISCI_WEB_SEARCH is off; web_search/link_summary will be removed from live prompts",
        )
    )
    checks.append(
        PaperDoctorCheck(
            "paper console",
            "ok",
            "CLI: aisci paper run / doctor / validate / resume; TUI: aisci tui and aisci tui job <id>.",
        )
    )
    return checks


def paper_job_summary(job: JobRecord, validation: dict[str, Any] | None = None) -> dict[str, Any]:
    job_paths = resolve_job_paths(job.id)
    workspace_root = job_paths.workspace_dir
    artifact_paths = []
    for candidate in (
        job_paths.artifacts_dir / "validation_report.json",
        job_paths.logs_dir / "paper_session_state.json",
        job_paths.state_dir / "sandbox_session.json",
        job_paths.state_dir / "resolved_llm_config.json",
        job_paths.logs_dir / "job.log",
    ):
        if candidate.exists():
            artifact_paths.append(str(candidate))
    return {
        "id": job.id,
        "job_type": job.job_type.value,
        "status": job.status.value,
        "phase": job.phase.value,
        "objective": job.objective,
        "llm_profile": job.llm_profile,
        "workspace_root": str(workspace_root),
        "artifacts": artifact_paths,
        "validation": validation or {},
        "paper_capabilities": paper_capability_flags(job),
        "paper_log_targets": [target.__dict__ for target in paper_log_targets(job)],
        "paper_artifacts": [hint.__dict__ for hint in paper_artifact_hints(job)],
    }


def job_console_summary(job: JobRecord, *, validation: dict[str, Any] | None = None) -> dict[str, Any]:
    job_paths = resolve_job_paths(job.id)
    tree_root = job_paths.root
    summary = paper_job_summary(job, validation=validation)
    summary["tree"] = list_text_tree(tree_root, max_depth=4, max_entries=300)
    summary["tree_root"] = str(tree_root)
    summary["workspace_tree"] = list_text_tree(job_paths.workspace_dir, max_depth=4, max_entries=200)
    summary["log_tree"] = list_text_tree(job_paths.logs_dir, max_depth=2, max_entries=50)
    summary["artifact_tree"] = list_text_tree(job_paths.artifacts_dir, max_depth=3, max_entries=150)
    summary["paper_log_targets"] = paper_log_targets(job)
    summary["paper_artifacts"] = paper_artifact_hints(job)
    summary["paper_capabilities"] = paper_capability_flags(job)
    return summary
