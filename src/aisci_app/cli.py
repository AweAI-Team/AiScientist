from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from aisci_app.service import JobService
from aisci_app.presentation import (
    build_job_spec_clone,
    build_mle_job_spec,
    build_paper_job_spec,
    default_mle_llm_profile_name,
    mle_doctor_report,
    paper_doctor_report,
)
from aisci_app.tui import run_mle_launcher, run_tui_dashboard
from aisci_agent_runtime.llm_profiles import default_llm_profile_name
from aisci_core.models import JobStatus, PullPolicy
from aisci_core.env_config import load_runtime_env
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_core.store import JobStore
from aisci_domain_mle.preflight import evaluate_mle_launch_preflight

app = typer.Typer(help="AI Scientist Workbench")
paper_app = typer.Typer(help="paper mode")
mle_app = typer.Typer(help="mle mode", invoke_without_command=True, no_args_is_help=False)
jobs_app = typer.Typer(help="inspect jobs")
logs_app = typer.Typer(help="inspect logs")
artifacts_app = typer.Typer(help="inspect artifacts")
tui_app = typer.Typer(help="live terminal dashboard", invoke_without_command=True, no_args_is_help=False)

app.add_typer(paper_app, name="paper")
app.add_typer(mle_app, name="mle")
app.add_typer(jobs_app, name="jobs")
app.add_typer(logs_app, name="logs")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(tui_app, name="tui")


@app.callback()
def main(
    env_file: Annotated[
        str | None,
        typer.Option(
            "--env-file",
            help="Load environment variables from this file before running the command. Defaults to .env/.env.aisci/.env.local in the repo root or current directory.",
        ),
    ] = None,
    output_root: Annotated[
        str | None,
        typer.Option(
            "--output-root",
            help="Write jobs/, export/, and .aisci state under this directory for this invocation.",
        ),
    ] = None,
    llm_profile_file: Annotated[
        str | None,
        typer.Option(
            "--llm-profile-file",
            help="Override the shared LLM profile registry file for this invocation.",
        ),
    ] = None,
    image_profile_file: Annotated[
        str | None,
        typer.Option(
            "--image-profile-file",
            help="Override the shared Docker image profile registry file for this invocation.",
        ),
    ] = None,
) -> None:
    load_runtime_env(env_file)
    if output_root:
        os.environ["AISCI_OUTPUT_ROOT"] = str(Path(output_root).expanduser().resolve())
    if llm_profile_file:
        os.environ["AISCI_LLM_PROFILE_FILE"] = str(Path(llm_profile_file).expanduser().resolve())
    if image_profile_file:
        os.environ["AISCI_IMAGE_PROFILE_FILE"] = str(Path(image_profile_file).expanduser().resolve())


def _print_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _parse_gpu_ids(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [item.strip() for item in raw.split(",")]
    gpu_ids = [item for item in values if item]
    if not gpu_ids:
        raise typer.BadParameter("--gpu-ids must contain at least one GPU id, e.g. --gpu-ids 4,5")
    return gpu_ids


def _get_job_or_exit(job_id: str):
    try:
        return JobStore().get_job(job_id)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _emit_job_launch_result(
    service: JobService,
    job_id: str,
    worker_value: int,
    *,
    wait: bool,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {"job_id": job_id, **(extra or {})}
    if not wait:
        payload.update({"worker": worker_value, "status": "started"})
        _print_json(payload)
        return

    job = service.store.get_job(job_id)
    payload.update(
        {
            "worker_exit_code": worker_value,
            "status": job.status.value,
            "phase": job.phase.value,
            "error": job.error,
        }
    )
    _print_json(payload)
    if worker_value != 0 or job.status != JobStatus.SUCCEEDED:
        raise typer.Exit(code=1)


def _ensure_mle_launch_ready(spec) -> None:
    preflight = evaluate_mle_launch_preflight(spec)
    if preflight.ready:
        return
    for warning in preflight.warnings:
        typer.echo(f"[warn] {warning}", err=True)
    for error in preflight.errors:
        typer.echo(error, err=True)
    raise typer.Exit(code=2)


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _run_tui_or_exit(
    *,
    job_id: str | None,
    refresh_seconds: float,
    once: bool,
    exit_when_job_done: bool = False,
    store: JobStore | None = None,
):
    try:
        return run_tui_dashboard(
            job_id=job_id,
            refresh_seconds=refresh_seconds,
            once=once,
            exit_when_job_done=exit_when_job_done,
            store=store,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _print_file_tail(label: str, path: Path, lines: int) -> bool:
    if not path.exists():
        typer.echo(f"[missing] {label}: {path}")
        return False
    if path.is_dir():
        typer.echo(f"[directory] {label}: {path}")
        children = sorted(path.rglob("*"))
        if not children:
            typer.echo("  (empty)")
        for child in children:
            if child.is_file():
                typer.echo(f"  - {child.relative_to(path)}")
        return True
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    typer.echo(f"[{label}] {path}")
    typer.echo("\n".join(content[-lines:]))
    typer.echo("")
    return True


def _log_targets_for_kind(job_id: str, kind: str, subagent: str | None = None) -> list[tuple[str, Path]]:
    paths = ensure_job_dirs(resolve_job_paths(job_id))
    kind = kind.strip().lower()
    targets: list[tuple[str, Path]] = []
    if kind in {"main", "all"}:
        targets.extend(
            [
                ("main job log", paths.logs_dir / "job.log"),
                ("worker log", paths.logs_dir / "worker.log"),
            ]
        )
    if kind in {"conversation", "all"}:
        targets.append(("conversation log", paths.logs_dir / "conversation.jsonl"))
    if kind in {"agent", "all"}:
        targets.append(("agent log", paths.logs_dir / "agent.log"))
    if kind in {"subagent", "subagents", "all"}:
        subagent_dir = paths.logs_dir / "subagent_logs"
        if subagent and subagent.strip():
            patterns = [f"*{subagent.strip()}*"]
            for pattern in patterns:
                for path in sorted(subagent_dir.glob(pattern)):
                    targets.append((f"subagent {path.name}", path))
        else:
            targets.append(("subagent logs", subagent_dir))
    if kind in {"validation", "all"}:
        targets.extend(
            [
                ("validation report", paths.artifacts_dir / "validation_report.json"),
                ("self-check report", paths.workspace_dir / "agent" / "final_self_check.md"),
                ("self-check report json", paths.workspace_dir / "agent" / "final_self_check.json"),
            ]
        )
    return targets


@paper_app.command("run")
def run_paper(
    pdf: Annotated[str | None, typer.Option("--pdf")] = None,
    paper_bundle_zip: Annotated[str | None, typer.Option("--paper-bundle-zip")] = None,
    paper_md: Annotated[str | None, typer.Option("--paper-md")] = None,
    llm_profile: Annotated[
        str | None,
        typer.Option("--llm-profile", help="Profile key from the LLM profiles YAML registry."),
    ] = None,
    gpus: Annotated[int, typer.Option("--gpus")] = 0,
    gpu_ids_raw: Annotated[
        str | None,
        typer.Option("--gpu-ids", help="Comma-separated GPU device ids, e.g. 4,5. Use this to pin specific GPUs."),
    ] = None,
    time_limit: Annotated[str, typer.Option("--time-limit")] = "24h",
    image: Annotated[str | None, typer.Option("--image", help="Docker image ref for the sandbox runtime.")] = None,
    pull_policy: Annotated[
        PullPolicy | None,
        typer.Option("--pull-policy", help="Image pull policy: if-missing, always, or never."),
    ] = None,
    inputs_zip: Annotated[str | None, typer.Option("--inputs-zip")] = None,
    rubric_path: Annotated[str | None, typer.Option("--rubric-path")] = None,
    blacklist_path: Annotated[str | None, typer.Option("--blacklist-path")] = None,
    addendum_path: Annotated[str | None, typer.Option("--addendum-path")] = None,
    seed_repo_zip: Annotated[str | None, typer.Option("--submission-seed-repo-zip")] = None,
    supporting_materials: Annotated[list[str] | None, typer.Option("--supporting-materials")] = None,
    run_final_validation: Annotated[bool, typer.Option("--run-final-validation/--skip-final-validation")] = True,
    detach: Annotated[bool, typer.Option("--detach/--wait")] = True,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help="Attach the live terminal dashboard after starting the job. Requires --wait and an interactive terminal.",
        ),
    ] = False,
) -> None:
    service = JobService()
    selected_llm_profile = llm_profile or default_llm_profile_name("paper")
    gpu_ids = _parse_gpu_ids(gpu_ids_raw)
    if gpu_ids and gpus > 0:
        raise typer.BadParameter("Use either --gpus <count> or --gpu-ids <id,id>, not both.")
    if tui and detach:
        raise typer.BadParameter("--tui requires --wait.")
    if tui and not _is_interactive_terminal():
        raise typer.BadParameter("--tui requires an interactive terminal.")
    spec = build_paper_job_spec(
        pdf_path=pdf,
        paper_bundle_zip=paper_bundle_zip,
        paper_md_path=paper_md,
        llm_profile=selected_llm_profile,
        gpus=gpus,
        gpu_ids=gpu_ids,
        time_limit=time_limit,
        inputs_zip=inputs_zip,
        rubric_path=rubric_path,
        blacklist_path=blacklist_path,
        addendum_path=addendum_path,
        seed_repo_zip=seed_repo_zip,
        supporting_materials=supporting_materials or [],
        image=image,
        pull_policy=pull_policy,
        run_final_validation=run_final_validation,
    )
    job = service.create_job(spec)
    wait = not detach
    if tui:
        worker_pid = service.spawn_worker(job.id, wait=False)
        result = _run_tui_or_exit(
            job_id=job.id,
            refresh_seconds=2.0,
            once=False,
            exit_when_job_done=True,
            store=service.store,
        )
        if result.detached:
            _emit_job_launch_result(service, job.id, worker_pid, wait=False)
            return
        final_job = service.store.get_job(job.id)
        if final_job.status != JobStatus.SUCCEEDED:
            raise typer.Exit(code=1)
        return
    worker_value = service.spawn_worker(job.id, wait=wait)
    _emit_job_launch_result(service, job.id, worker_value, wait=wait)


@paper_app.command("doctor")
def paper_doctor(json_output: Annotated[bool, typer.Option("--json/--text")] = False) -> None:
    checks = paper_doctor_report()
    if json_output:
        _print_json([check.__dict__ for check in checks])
        return
    typer.echo("Paper Doctor")
    for check in checks:
        typer.echo(f"- {check.name}: {check.status} ({check.detail})")
    typer.echo("")
    typer.echo("Start a paper job with:")
    typer.echo("  aisci paper run --pdf /path/to/paper.pdf --wait --tui")


@paper_app.command("validate")
def paper_validate(
    job_id: str,
    detach: Annotated[bool, typer.Option("--detach/--wait")] = True,
) -> None:
    job = _get_job_or_exit(job_id)
    spec = build_job_spec_clone(job, objective_suffix=" [self-check]", run_final_validation=True)
    service = JobService()
    new_job = service.create_job(spec)
    wait = not detach
    worker_value = service.spawn_worker(new_job.id, wait=wait)
    _emit_job_launch_result(
        service,
        new_job.id,
        worker_value,
        wait=wait,
        extra={"source_job_id": job_id, "mode": "self-check"},
    )


@paper_app.command("resume")
def paper_resume(
    job_id: str,
    detach: Annotated[bool, typer.Option("--detach/--wait")] = True,
) -> None:
    job = _get_job_or_exit(job_id)
    spec = build_job_spec_clone(job, objective_suffix=" [resumed]")
    service = JobService()
    new_job = service.create_job(spec)
    wait = not detach
    worker_value = service.spawn_worker(new_job.id, wait=wait)
    _emit_job_launch_result(
        service,
        new_job.id,
        worker_value,
        wait=wait,
        extra={"source_job_id": job_id, "mode": "resume"},
    )


@mle_app.command("run")
def run_mle(
    ctx: typer.Context,
    competition_name: Annotated[str | None, typer.Option("--name")] = None,
    competition_zip_path: Annotated[str | None, typer.Option("--zip")] = None,
    mlebench_data_dir: Annotated[
        str | None,
        typer.Option(
            "--mlebench-data-dir",
            help="Prepared MLE-Bench cache root. Defaults to ~/.cache/mle-bench/data when omitted.",
        ),
    ] = None,
    workspace_zip: Annotated[str | None, typer.Option("--workspace-zip")] = None,
    competition_bundle_zip: Annotated[str | None, typer.Option("--competition-bundle-zip")] = None,
    data_dir: Annotated[str | None, typer.Option("--data-dir")] = None,
    code_repo_zip: Annotated[str | None, typer.Option("--code-repo-zip")] = None,
    description_path: Annotated[str | None, typer.Option("--description-path")] = None,
    sample_submission_path: Annotated[str | None, typer.Option("--sample-submission-path")] = None,
    validation_command: Annotated[str | None, typer.Option("--validation-command")] = None,
    grading_config_path: Annotated[str | None, typer.Option("--grading-config-path")] = None,
    metric_direction: Annotated[str | None, typer.Option("--metric-direction")] = None,
    llm_profile: Annotated[
        str | None,
        typer.Option("--llm-profile", help="Profile key from the LLM profiles YAML registry."),
    ] = None,
    gpus: Annotated[int, typer.Option("--gpus")] = 0,
    gpu_ids_raw: Annotated[
        str | None,
        typer.Option("--gpu-ids", help="Comma-separated GPU device ids, e.g. 4,5. Use this to pin specific GPUs."),
    ] = None,
    time_limit: Annotated[str, typer.Option("--time-limit")] = "24h",
    image: Annotated[str | None, typer.Option("--image", help="Docker image ref for the sandbox runtime.")] = None,
    pull_policy: Annotated[
        PullPolicy | None,
        typer.Option("--pull-policy", help="Image pull policy: if-missing, always, or never."),
    ] = None,
    run_final_validation: Annotated[
        bool,
        typer.Option("--run-final-validation/--skip-final-validation"),
    ] = False,
    detach: Annotated[bool, typer.Option("--detach/--wait")] = True,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help="Attach the live terminal dashboard after starting the job. Requires --wait and an interactive terminal.",
        ),
    ] = False,
) -> None:
    service = JobService()
    selected_llm_profile = llm_profile or default_mle_llm_profile_name()
    gpu_ids = _parse_gpu_ids(gpu_ids_raw)
    inherited_tui = bool(getattr(getattr(ctx, "parent", None), "params", {}).get("tui"))
    effective_tui = tui or inherited_tui
    wait = not detach or inherited_tui
    if gpu_ids and gpus > 0:
        raise typer.BadParameter("Use either --gpus <count> or --gpu-ids <id,id>, not both.")
    if effective_tui and wait is False:
        raise typer.BadParameter("--tui requires --wait.")
    if effective_tui and not _is_interactive_terminal():
        raise typer.BadParameter("--tui requires an interactive terminal.")
    spec = build_mle_job_spec(
        competition_name=competition_name,
        competition_zip_path=competition_zip_path,
        mlebench_data_dir=mlebench_data_dir,
        workspace_zip=workspace_zip,
        competition_bundle_zip=competition_bundle_zip,
        data_dir=data_dir,
        code_repo_zip=code_repo_zip,
        description_path=description_path,
        sample_submission_path=sample_submission_path,
        validation_command=validation_command,
        grading_config_path=grading_config_path,
        metric_direction=metric_direction,
        llm_profile=selected_llm_profile,
        gpus=gpus,
        gpu_ids=gpu_ids,
        time_limit=time_limit,
        image=image,
        pull_policy=pull_policy,
        run_final_validation=run_final_validation,
    )
    _ensure_mle_launch_ready(spec)
    job = service.create_job(spec)
    if effective_tui:
        worker_pid = service.spawn_worker(job.id, wait=False)
        result = _run_tui_or_exit(
            job_id=job.id,
            refresh_seconds=2.0,
            once=False,
            exit_when_job_done=True,
            store=service.store,
        )
        if result.detached:
            _emit_job_launch_result(service, job.id, worker_pid, wait=False)
            return
        final_job = service.store.get_job(job.id)
        if final_job.status != JobStatus.SUCCEEDED:
            raise typer.Exit(code=1)
        return
    worker_value = service.spawn_worker(job.id, wait=wait)
    _emit_job_launch_result(service, job.id, worker_value, wait=wait)


@mle_app.callback()
def mle_root(
    ctx: typer.Context,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help="Open the interactive MLE launcher and attach the live dashboard after starting a job.",
        ),
    ] = False,
) -> None:
    if ctx.invoked_subcommand is not None:
        if tui and ctx.invoked_subcommand != "run":
            raise typer.BadParameter(
                "--tui on `aisci mle` is only supported with `run` or with no subcommand. "
                "Use `aisci mle run ... --wait --tui` or `aisci mle --tui`."
            )
        return
    if tui:
        if not _is_interactive_terminal():
            raise typer.BadParameter("--tui requires an interactive terminal.")
        store = JobStore()
        result = run_mle_launcher(store=store)
        if result is None or result.job_id is None:
            return
        if result.detached:
            return
        final_job = store.get_job(result.job_id)
        if final_job.status != JobStatus.SUCCEEDED:
            raise typer.Exit(code=1)
        return
    typer.echo(ctx.get_help())
    raise typer.Exit()


@mle_app.command("doctor")
def mle_doctor(json_output: Annotated[bool, typer.Option("--json/--text")] = False) -> None:
    checks = mle_doctor_report()
    if json_output:
        _print_json([check.__dict__ for check in checks])
        return
    typer.echo("MLE Doctor")
    for check in checks:
        typer.echo(f"- {check.name}: {check.status} ({check.detail})")
    typer.echo("")
    typer.echo("Start an MLE job with:")
    typer.echo("  aisci mle run --name detecting-insults-in-social-commentary")


@mle_app.command("validate")
def mle_validate(
    job_id: str,
    detach: Annotated[bool, typer.Option("--detach/--wait")] = True,
) -> None:
    job = _get_job_or_exit(job_id)
    spec = build_job_spec_clone(job, objective_suffix=" [self-check]", run_final_validation=True)
    _ensure_mle_launch_ready(spec)
    service = JobService()
    new_job = service.create_job(spec)
    wait = not detach
    worker_value = service.spawn_worker(new_job.id, wait=wait)
    _emit_job_launch_result(
        service,
        new_job.id,
        worker_value,
        wait=wait,
        extra={"source_job_id": job_id, "mode": "self-check"},
    )


@mle_app.command("resume")
def mle_resume(
    job_id: str,
    detach: Annotated[bool, typer.Option("--detach/--wait")] = True,
) -> None:
    job = _get_job_or_exit(job_id)
    spec = build_job_spec_clone(job, objective_suffix=" [resumed]")
    _ensure_mle_launch_ready(spec)
    service = JobService()
    new_job = service.create_job(spec)
    wait = not detach
    worker_value = service.spawn_worker(new_job.id, wait=wait)
    _emit_job_launch_result(
        service,
        new_job.id,
        worker_value,
        wait=wait,
        extra={"source_job_id": job_id, "mode": "resume"},
    )


@jobs_app.command("list")
def jobs_list() -> None:
    store = JobStore()
    jobs = [job.model_dump(mode="json") for job in store.list_jobs()]
    _print_json(jobs)


@jobs_app.command("show")
def jobs_show(job_id: str) -> None:
    store = JobStore()
    job = _get_job_or_exit(job_id)
    payload = job.model_dump(mode="json")
    payload["events"] = [event.model_dump(mode="json") for event in store.list_events(job_id)]
    payload["artifacts"] = [artifact.model_dump(mode="json") for artifact in store.list_artifacts(job_id)]
    _print_json(payload)


@logs_app.command("tail")
def logs_tail(
    job_id: str,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="Which log family to show: main, conversation, agent, subagent, validation, all",
        ),
    ] = "main",
    lines: Annotated[int, typer.Option("--lines")] = 40,
    subagent: Annotated[str | None, typer.Option("--subagent")] = None,
) -> None:
    targets = _log_targets_for_kind(job_id, kind, subagent=subagent)
    shown = 0
    for label, path in targets:
        shown += int(_print_file_tail(label, path, lines))
    if shown == 0:
        typer.echo(f"No logs found for job {job_id} and kind={kind}", err=True)
        raise typer.Exit(code=1)


@logs_app.command("list")
def logs_list(job_id: str) -> None:
    targets = _log_targets_for_kind(job_id, "all")
    payload = [
        {"label": label, "path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}
        for label, path in targets
    ]
    _print_json(payload)


@artifacts_app.command("ls")
def artifacts_ls(job_id: str) -> None:
    store = JobStore()
    payload = [artifact.model_dump(mode="json") for artifact in store.list_artifacts(job_id)]
    _print_json(payload)


@tui_app.callback()
def tui_root(
    ctx: typer.Context,
    refresh: Annotated[float, typer.Option("--refresh", help="Refresh interval in seconds.")] = 2.0,
    once: Annotated[bool, typer.Option("--once", help="Render a single frame and exit.")] = False,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    _run_tui_or_exit(job_id=None, refresh_seconds=refresh, once=once)


@tui_app.command("job")
def tui_job(
    job_id: str,
    refresh: Annotated[float, typer.Option("--refresh", help="Refresh interval in seconds.")] = 2.0,
    once: Annotated[bool, typer.Option("--once", help="Render a single frame and exit.")] = False,
) -> None:
    _get_job_or_exit(job_id)
    _run_tui_or_exit(job_id=job_id, refresh_seconds=refresh, once=once)


@app.command("export")
def export_job(job_id: str) -> None:
    path = JobService().export_bundle(job_id)
    typer.echo(str(path))
