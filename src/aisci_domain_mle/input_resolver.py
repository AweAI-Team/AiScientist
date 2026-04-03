from __future__ import annotations

import os
from pathlib import Path

from aisci_domain_mle.contracts import (
    DryRunStatus,
    InputSourceKind,
    MLEPhase1JobSpec,
    MLEPhase1ModeSpec,
    MLEPhase1RuntimeSpec,
    Phase1DryRunReport,
    ResolvedInputState,
)
from aisci_domain_mle.mlebench_compat import resolve_competition_source


def normalize_gpu_ids(raw_value: str | int | None) -> list[str]:
    if raw_value is None:
        return []
    raw_text = str(raw_value).strip()
    if not raw_text:
        return []
    gpu_ids: list[str] = []
    seen: set[str] = set()
    for chunk in raw_text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if not item.isdigit():
            raise ValueError(f"invalid gpu id {item!r}; expected comma-separated integers such as 0,1")
        if item not in seen:
            seen.add(item)
            gpu_ids.append(item)
    return gpu_ids


def default_mlebench_data_dir() -> Path:
    configured = os.environ.get("MLEBENCH_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "mle-bench" / "data").resolve()


def infer_competition_name(name: str | None, zip_path: str | None) -> str | None:
    cleaned = (name or "").strip()
    if cleaned:
        return cleaned
    if not zip_path:
        return None
    zip_name = Path(zip_path).name
    if zip_name.lower().endswith(".zip"):
        zip_name = zip_name[:-4]
    return zip_name or None


def _normalize_optional_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    return str(Path(raw_path).expanduser().resolve())


def _pick_source_kind(mode_spec: MLEPhase1ModeSpec) -> InputSourceKind:
    if mode_spec.competition_zip_path:
        return InputSourceKind.LOCAL_ZIP
    if mode_spec.data_dir:
        return InputSourceKind.DATA_DIR
    if mode_spec.workspace_bundle_zip:
        return InputSourceKind.WORKSPACE_BUNDLE
    if mode_spec.competition_bundle_zip:
        return InputSourceKind.COMPETITION_BUNDLE
    return InputSourceKind.COMPETITION_NAME


def _safe_public_data_dir(source: Path) -> Path | None:
    source = source.expanduser().resolve()
    if source.name == "public" and source.is_dir():
        return source
    if source.name == "prepared" and (source / "public").is_dir():
        return (source / "public").resolve()
    if (source / "prepared" / "public").is_dir():
        return (source / "prepared" / "public").resolve()
    return None


def _has_api_key() -> bool:
    return any(os.environ.get(name) for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"))


def build_phase1_job_spec(
    *,
    competition_name: str | None,
    competition_zip_path: str | None,
    mlebench_data_dir: str | None,
    workspace_bundle_zip: str | None,
    competition_bundle_zip: str | None,
    data_dir: str | None,
    code_repo_zip: str | None,
    description_path: str | None,
    sample_submission_path: str | None,
    validation_command: str | None,
    grading_config_path: str | None,
    metric_direction: str | None,
    llm_profile: str,
    gpus: str | int | None,
    time_limit: str,
    dockerfile: str | None,
    run_final_validation: bool,
    dry_run: bool,
    objective: str,
) -> MLEPhase1JobSpec:
    inferred_name = infer_competition_name(competition_name, competition_zip_path)
    mode_spec = MLEPhase1ModeSpec(
        competition_name=inferred_name,
        competition_zip_path=_normalize_optional_path(competition_zip_path),
        mlebench_data_dir=_normalize_optional_path(mlebench_data_dir)
        if mlebench_data_dir
        else str(default_mlebench_data_dir()),
        workspace_bundle_zip=_normalize_optional_path(workspace_bundle_zip),
        competition_bundle_zip=_normalize_optional_path(competition_bundle_zip),
        data_dir=_normalize_optional_path(data_dir),
        code_repo_zip=_normalize_optional_path(code_repo_zip),
        description_path=_normalize_optional_path(description_path),
        sample_submission_path=_normalize_optional_path(sample_submission_path),
        validation_command=validation_command,
        grading_config_path=_normalize_optional_path(grading_config_path),
        metric_direction=metric_direction,
    )
    runtime_profile = MLEPhase1RuntimeSpec(
        gpu_ids=normalize_gpu_ids(gpus),
        time_limit=(time_limit or "24h").strip() or "24h",
        dockerfile_path=_normalize_optional_path(dockerfile),
        run_final_validation=run_final_validation,
        dry_run=dry_run,
    )
    return MLEPhase1JobSpec(
        objective=(objective or "mle optimization job").strip() or "mle optimization job",
        llm_profile=(llm_profile or "gpt-5.4").strip() or "gpt-5.4",
        runtime_profile=runtime_profile,
        mode_spec=mode_spec,
    )


def build_dry_run_report(job_spec: MLEPhase1JobSpec, *, wait_requested: bool) -> Phase1DryRunReport:
    mode_spec = job_spec.mode_spec
    cache_root = Path(mode_spec.mlebench_data_dir or default_mlebench_data_dir()).expanduser().resolve()
    warnings: list[str] = []
    source_kind = _pick_source_kind(mode_spec)
    if source_kind in {InputSourceKind.LOCAL_ZIP, InputSourceKind.COMPETITION_NAME}:
        resolved_inputs = resolve_competition_source(
            competition_name=mode_spec.competition_name,
            competition_zip_path=mode_spec.competition_zip_path,
            cache_root=cache_root,
            allow_download=False,
        )
        zip_path = (
            Path(resolved_inputs.competition_zip_path).resolve()
            if resolved_inputs.competition_zip_path
            else None
        )
        competition_name = resolved_inputs.competition_name

        if zip_path and zip_path.suffix.lower() != ".zip":
            warnings.append(f"competition zip path does not end with .zip: {zip_path}")
        if zip_path and competition_name and zip_path.stem != competition_name:
            warnings.append(
                f"zip stem {zip_path.stem!r} differs from competition_name {competition_name!r}; "
                "phase2 preparation should use competition_name as the registry id"
            )
        if source_kind == InputSourceKind.COMPETITION_NAME and not resolved_inputs.cache_prepared_exists:
            warnings.append(
                "prepared cache is missing for this competition name; the legacy MLE-Bench prepare path is wired "
                "and exposed in resolved_inputs.legacy_prepare_plan"
            )
    else:
        resolved_inputs = ResolvedInputState(
            source_kind=source_kind,
            cache_root=str(cache_root),
        )
        if source_kind in {InputSourceKind.WORKSPACE_BUNDLE, InputSourceKind.COMPETITION_BUNDLE}:
            warnings.append(
                f"{source_kind.value} input is still a migration helper path; prefer the shared `aisci mle run --name/--zip` entrypoint for authoritative live behavior."
            )

    if source_kind == InputSourceKind.LOCAL_ZIP:
        next_steps = [
            "Live MLE runs prepare the local zip into temporary public/private data before the shared container session starts.",
            "Use --name together with --zip when the archive filename does not already match the legacy competition slug.",
        ]
        status = DryRunStatus.READY
    elif source_kind == InputSourceKind.COMPETITION_NAME and resolved_inputs.cache_prepared_exists:
        next_steps = [
            "The live adapter can stage public data directly from the prepared cache.",
            "Cache-hit competition-name runs are ready once required public metadata can be staged.",
        ]
        status = DryRunStatus.READY
    elif source_kind == InputSourceKind.DATA_DIR:
        if mode_spec.data_dir and _safe_public_data_dir(Path(mode_spec.data_dir)):
            next_steps = [
                "Provide a public competition directory, a prepared directory, or a competition root containing prepared/public.",
                "The live adapter will stage only the public subtree into the solver workspace.",
            ]
            status = DryRunStatus.READY
        else:
            warnings.append(
                "data_dir must point to a public competition directory, a prepared directory, or a competition root containing prepared/public."
            )
            next_steps = [
                "Fix --data-dir so it resolves to a safe public competition tree before launching the live adapter.",
                "If you only have a raw bundle, use --zip so the live adapter can prepare public/private staging first.",
            ]
            status = DryRunStatus.NEEDS_PREPARE
    else:
        next_steps = [
            "Provide --zip for an offline path, or run the wired legacy MLE-Bench prepare command before full execution.",
            "If networked preparation is needed later, the operator must run proxy-on first.",
        ]
        status = DryRunStatus.NEEDS_PREPARE

    return Phase1DryRunReport(
        status=status,
        wait_requested=wait_requested,
        api_key_required=False,
        api_key_present=_has_api_key(),
        job_spec=job_spec,
        resolved_inputs=resolved_inputs,
        warnings=warnings,
        next_steps=next_steps,
    )
