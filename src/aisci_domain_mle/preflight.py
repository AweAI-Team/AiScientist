from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path

from aisci_core.models import JobSpec, JobType, MLESpec, PullPolicy
from aisci_domain_mle.mlebench_compat import LegacyCompetitionPreparer, resolve_competition_source
from aisci_domain_mle.shared_infra_bridge import default_domain_mle_profile, present_proxy_env_keys
from aisci_runtime_docker.runtime import DockerRuntimeManager

REQUIRED_PUBLIC_METADATA_FILENAMES = ("description.md", "sample_submission.csv")
ORCHESTRATOR_FINALIZE_BUFFER_SECONDS = 120
ORCHESTRATOR_KILL_AFTER_SECONDS = 30
HOST_EXEC_TIMEOUT_BUFFER_SECONDS = ORCHESTRATOR_KILL_AFTER_SECONDS + 15


@dataclass(frozen=True)
class MLELaunchPreflight:
    ready: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        if self.errors:
            return self.errors[0]
        if self.warnings:
            return self.warnings[0]
        return "MLE launch preflight passed."


def orchestrator_shell_timeout_seconds(time_limit_seconds: int) -> int:
    return max(1, time_limit_seconds) + ORCHESTRATOR_FINALIZE_BUFFER_SECONDS


def orchestrator_host_timeout_seconds(time_limit_seconds: int) -> int:
    return (
        orchestrator_shell_timeout_seconds(time_limit_seconds)
        + HOST_EXEC_TIMEOUT_BUFFER_SECONDS
    )


def orchestrator_exec_command(time_limit_seconds: int) -> str:
    shell_timeout = orchestrator_shell_timeout_seconds(time_limit_seconds)
    return (
        "timeout --signal=TERM "
        f"--kill-after={ORCHESTRATOR_KILL_AFTER_SECONDS} "
        f"{shell_timeout} "
        "python -m aisci_domain_mle.orchestrator"
    )


def restore_submission_from_workspace(workspace_dir: Path) -> Path | None:
    submission_path = workspace_dir / "submission" / "submission.csv"
    if submission_path.is_file():
        return submission_path

    candidates = (
        workspace_dir / "code" / "submission.csv",
        workspace_dir / "code" / "output" / "submission.csv",
        workspace_dir / "code" / "submissions" / "submission.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, submission_path)
            return submission_path
    return None


def validate_required_public_metadata(data_dir: Path) -> None:
    missing = [
        name
        for name in REQUIRED_PUBLIC_METADATA_FILENAMES
        if not (data_dir / name).is_file()
    ]
    if not missing:
        return
    missing_list = ", ".join(missing)
    raise ValueError(
        "MLE runs require public competition metadata before the solve loop starts; "
        f"missing {missing_list} under {data_dir}. "
        "Provide safe public overrides with --description-path/--sample-submission-path "
        "or ensure the legacy competition metadata is available."
    )


def proxy_enabled() -> bool:
    return bool(present_proxy_env_keys())


def preflight_doctor_warnings(runtime: DockerRuntimeManager | None = None) -> tuple[str, ...]:
    runtime = runtime or DockerRuntimeManager()
    warnings: list[str] = []
    profile = default_domain_mle_profile()
    if profile.pull_policy == PullPolicy.ALWAYS:
        warnings.append(
            "Default MLE runtime image uses pull_policy=always; live runs require proxy-on before image pulls."
        )
    elif profile.pull_policy == PullPolicy.IF_MISSING and runtime.can_use_docker():
        if not runtime.image_exists(profile.image):
            warnings.append(
                "Default MLE runtime image is not present locally; the first live run will need proxy-on before Docker pulls it."
            )
    warnings.append(
        "Competition-name runs require prepared cache under ~/.cache/mle-bench/data/<competition>; cache misses are blocked until the operator runs proxy-on and the built-in MLE prepare command."
    )
    return tuple(warnings)


def evaluate_mle_launch_preflight(
    job_spec: JobSpec,
    *,
    runtime: DockerRuntimeManager | None = None,
    competition_preparer: LegacyCompetitionPreparer | None = None,
) -> MLELaunchPreflight:
    if job_spec.job_type != JobType.MLE or not isinstance(job_spec.mode_spec, MLESpec):
        raise ValueError("MLE preflight requires an MLE JobSpec")

    runtime = runtime or DockerRuntimeManager()
    competition_preparer = competition_preparer or LegacyCompetitionPreparer()
    spec = job_spec.mode_spec
    errors: list[str] = []
    warnings: list[str] = []

    if not runtime.can_use_docker():
        errors.append(
            "MLE mode requires a reachable Docker daemon. No local fallback loop is available."
        )
        return MLELaunchPreflight(ready=False, errors=tuple(errors), warnings=tuple(warnings))

    profile = default_domain_mle_profile()
    image_ref = (job_spec.runtime_profile.image or "").strip() or profile.image
    pull_policy = job_spec.runtime_profile.pull_policy or profile.pull_policy
    if pull_policy == PullPolicy.ALWAYS:
        if proxy_enabled():
            warnings.append(
                f"Runtime image {image_ref} will be pulled for this run; proxy environment is already present."
            )
        else:
            errors.append(
                f"Runtime image {image_ref} will be pulled for this run. Run `proxy-on` first, then retry `aisci mle run`."
            )
    elif pull_policy == PullPolicy.IF_MISSING and not runtime.image_exists(image_ref):
        if proxy_enabled():
            warnings.append(
                f"Runtime image {image_ref} is missing locally and will be pulled; proxy environment is already present."
            )
        else:
            errors.append(
                f"Runtime image {image_ref} is missing locally and this run would pull it. Run `proxy-on` first, then retry `aisci mle run`."
            )

    competition_name = getattr(spec, "competition_name", None)
    competition_zip_path = getattr(spec, "competition_zip_path", None)
    if competition_name and not competition_zip_path:
        resolved_inputs = resolve_competition_source(
            competition_name=competition_name,
            competition_zip_path=None,
            cache_root=_mlebench_cache_root(spec),
            allow_download=False,
        )
        if not resolved_inputs.cache_prepared_exists:
            prepare_command = (
                " ".join(resolved_inputs.legacy_prepare_plan.command)
                if resolved_inputs.legacy_prepare_plan
                else "python -m aisci_domain_mle.vendored_lite_cli prepare ..."
            )
            errors.append(
                "Prepared MLE-Bench cache is missing for competition "
                f"{competition_name!r}. Run `proxy-on` first, then `{prepare_command}`, then retry `aisci mle run`."
            )
        elif resolved_inputs.cache_prepared_dir:
            prepared_dir = Path(resolved_inputs.cache_prepared_dir).resolve()
            metadata_description = (
                Path(spec.description_path).resolve()
                if spec.description_path
                else None
            )
            metadata_sample = (
                Path(spec.sample_submission_path).resolve()
                if spec.sample_submission_path
                else None
            )
            if metadata_description is None or metadata_sample is None:
                try:
                    legacy_description, legacy_sample = competition_preparer.resolve_public_metadata_paths(
                        competition_name,
                        prepared_dir=prepared_dir,
                    )
                except ValueError:
                    legacy_description = None
                    legacy_sample = None
                metadata_description = metadata_description or legacy_description
                metadata_sample = metadata_sample or legacy_sample
            if metadata_description is None or not metadata_description.is_file():
                errors.append(
                    "Competition description metadata is missing for this prepared cache. "
                    "Provide --description-path or repair the legacy competition metadata before launching the run."
                )
            if metadata_sample is None or not metadata_sample.is_file():
                errors.append(
                    "Sample submission metadata is missing for this prepared cache. "
                    "Provide --sample-submission-path or repair the legacy competition metadata before launching the run."
                )

    return MLELaunchPreflight(
        ready=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _mlebench_cache_root(spec: MLESpec) -> Path:
    configured = getattr(spec, "mlebench_data_dir", None)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "mle-bench" / "data").resolve()
