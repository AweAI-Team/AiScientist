from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import quote
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aisci_agent_runtime.llm_profiles import default_llm_profile_name
from aisci_app.presentation import (
    build_job_spec_clone,
    build_mle_job_spec,
    build_paper_job_spec,
    job_console_summary,
    list_text_tree,
    paper_doctor_report,
    read_text_preview,
)
from aisci_app.service import JobService
from aisci_core.env_config import load_runtime_env
from aisci_core.models import JobRecord, JobSpec, JobType, MLESpec, NetworkPolicy, PaperSpec, PullPolicy, RuntimeProfile, WorkspaceLayout
from aisci_core.paths import ensure_job_dirs, repo_root, resolve_job_paths
from aisci_core.store import JobStore


def _to_relative(root: Path, candidate: Path) -> str:
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except Exception:
        return str(candidate)


def _within_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    items: list[str] = []
    for chunk in value.replace("\n", ",").split(","):
        item = chunk.strip()
        if item:
            items.append(item)
    return items


def _load_json_file(path: Path) -> Any | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _pick_default_preview(job_id: str) -> tuple[Path | None, str | None]:
    paths = ensure_job_dirs(resolve_job_paths(job_id))
    candidates = [
        paths.logs_dir / "job.log",
        paths.logs_dir / "worker.log",
        paths.state_dir / "resolved_llm_config.json",
        paths.artifacts_dir / "validation_report.json",
        paths.logs_dir / "paper_session_state.json",
        paths.state_dir / "sandbox_session.json",
        paths.workspace_dir / "agent" / "paper_analysis" / "summary.md",
        paths.workspace_dir / "agent" / "prioritized_tasks.md",
        paths.workspace_dir / "agent" / "impl_log.md",
        paths.workspace_dir / "agent" / "exp_log.md",
        paths.workspace_dir / "agent" / "final_self_check.md",
        paths.workspace_dir / "submission" / "reproduce.sh",
        paths.workspace_dir / "submission" / "README.md",
        paths.workspace_dir / "submission" / "submission.py",
        paths.workspace_dir / "submission" / "submission.csv",
        paths.workspace_dir / "submission" / "main.py",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate, _to_relative(paths.root, candidate)
    for node in list_text_tree(paths.root, max_depth=3, max_entries=120):
        candidate = paths.root / node["relative_path"]
        if candidate.exists() and candidate.is_file():
            return candidate, _to_relative(paths.root, candidate)
    return None, None


def _resolve_preview_path(job_id: str, preview: str | None) -> tuple[Path | None, str | None]:
    paths = ensure_job_dirs(resolve_job_paths(job_id))
    if preview:
        candidate = Path(preview)
        candidate = candidate if candidate.is_absolute() else (paths.root / candidate)
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_file() and _within_root(paths.root, candidate):
            return candidate, _to_relative(paths.root, candidate)
    return _pick_default_preview(job_id)


def _build_paper_form_spec(form: dict[str, Any]) -> JobSpec:
    runtime = RuntimeProfile(
        gpu_count=form["gpus"],
        time_limit=form["time_limit"],
        image=form["image"],
        pull_policy=PullPolicy(form["pull_policy"]) if form.get("pull_policy") else None,
        run_final_validation=form["run_final_validation"],
        network_policy=NetworkPolicy.BRIDGE,
        workspace_layout=WorkspaceLayout.PAPER,
    )
    return JobSpec(
        job_type=JobType.PAPER,
        objective=form["objective"],
        llm_profile=form["llm_profile"],
        runtime_profile=runtime,
        mode_spec=PaperSpec(
            pdf_path=form["pdf_path"],
            paper_bundle_zip=form["paper_bundle_zip"],
            paper_md_path=form["paper_md_path"],
            context_bundle_zip=form["inputs_zip"],
            rubric_path=form["rubric_path"],
            blacklist_path=form["blacklist_path"],
            addendum_path=form["addendum_path"],
            supporting_materials=form["supporting_materials"],
            submission_seed_repo_zip=form["submission_seed_repo_zip"],
            enable_online_research=form["enable_online_research"],
        ),
    )


def _build_mle_form_spec(form: dict[str, Any]) -> JobSpec:
    runtime = RuntimeProfile(
        gpu_count=form["gpus"],
        time_limit=form["time_limit"],
        image=form["image"],
        pull_policy=PullPolicy(form["pull_policy"]) if form.get("pull_policy") else None,
        run_final_validation=form["run_final_validation"],
        network_policy=NetworkPolicy.BRIDGE,
        workspace_layout=WorkspaceLayout.MLE,
    )
    return JobSpec(
        job_type=JobType.MLE,
        objective=form["objective"],
        llm_profile=form["llm_profile"],
        runtime_profile=runtime,
        mode_spec=MLESpec(
            workspace_bundle_zip=form["workspace_zip"],
            competition_bundle_zip=form["competition_bundle_zip"],
            data_dir=form["data_dir"],
            code_repo_zip=form["code_repo_zip"],
            description_path=form["description_path"],
            sample_submission_path=form["sample_submission_path"],
            validation_command=form["validation_command"],
            grading_config_path=form["grading_config_path"],
            metric_direction=form["metric_direction"],
        ),
    )


def _job_artifact_rows(job_id: str) -> list[dict[str, Any]]:
    store = JobStore()
    paths = ensure_job_dirs(resolve_job_paths(job_id))
    rows: list[dict[str, Any]] = []
    for artifact in store.list_artifacts(job_id):
        artifact_path = Path(artifact.path)
        if not artifact_path.is_absolute():
            artifact_path = (paths.root / artifact_path).resolve()
        rows.append(
            {
                "record": artifact,
                "path": str(artifact_path),
                "relative_path": _to_relative(paths.root, artifact_path),
                "exists": artifact_path.exists(),
                "is_text": artifact_path.suffix.lower() in {".md", ".txt", ".json", ".jsonl", ".log", ".sh"},
            }
        )
    return rows


def _augment_tree(job_id: str, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in nodes:
        rel = node["relative_path"]
        preview_href = None
        if rel != "." and not node["is_dir"]:
            preview_href = f"/jobs/{job_id}?preview={quote(rel)}"
        rows.append({**node, "preview_href": preview_href})
    return rows


def _job_view_context(job: JobRecord, *, preview: str | None = None) -> dict[str, Any]:
    paths = ensure_job_dirs(resolve_job_paths(job.id))
    store = JobStore()
    events = [event.model_dump(mode="json") for event in store.list_events(job.id)]
    artifacts = _job_artifact_rows(job.id)
    preview_path, preview_rel = _resolve_preview_path(job.id, preview)
    preview_payload = read_text_preview(preview_path) if preview_path else {"path": None, "exists": False, "kind": "missing", "content": ""}
    validation_report = None
    for row in artifacts:
        if row["relative_path"].endswith("validation_report.json"):
            validation_report = _load_json_file(Path(row["path"]))
            break
    return {
        "job": job,
        "job_json": job.model_dump(mode="json"),
        "events": events,
        "artifacts": artifacts,
        "preview_path": str(preview_path) if preview_path else None,
        "preview_rel": preview_rel,
        "preview": preview_payload
        if preview_path is not None and preview_payload.get("kind") != "missing"
        else {
            **preview_payload,
            "content": "Select a file from the tree or artifact list to inspect it here.",
        },
        "validation_report": validation_report,
        "console": job_console_summary(job, validation=validation_report),
        "tree": _augment_tree(job.id, list_text_tree(paths.root, max_depth=4, max_entries=300)),
        "workspace_tree": _augment_tree(job.id, list_text_tree(paths.workspace_dir, max_depth=4, max_entries=220)),
        "log_tree": _augment_tree(job.id, list_text_tree(paths.logs_dir, max_depth=3, max_entries=80)),
        "artifact_tree": _augment_tree(job.id, list_text_tree(paths.artifacts_dir, max_depth=3, max_entries=160)),
        "job_paths": paths,
    }


def create_app() -> FastAPI:
    load_runtime_env()
    app = FastAPI(title="AiScientist Workbench")
    templates = Jinja2Templates(directory=str(repo_root() / "src" / "aisci_app" / "templates"))
    static_dir = repo_root() / "src" / "aisci_app" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        store = JobStore()
        jobs = store.list_jobs()
        report = paper_doctor_report()
        return templates.TemplateResponse(
            "jobs.html",
            {
                "request": request,
                "jobs": jobs,
                "doctor_report": report,
                "default_pdf": "",
                "default_paper_llm_profile": default_llm_profile_name("paper"),
            },
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(request: Request, job_id: str, preview: str | None = None) -> HTMLResponse:
        store = JobStore()
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return templates.TemplateResponse("job_detail.html", {"request": request, **_job_view_context(job, preview=preview)})

    @app.post("/paper/jobs")
    async def create_paper_job(
        pdf_path: str | None = Form(None),
        paper_bundle_zip: str | None = Form(None),
        paper_md_path: str | None = Form(None),
        inputs_zip: str | None = Form(None),
        rubric_path: str | None = Form(None),
        blacklist_path: str | None = Form(None),
        addendum_path: str | None = Form(None),
        submission_seed_repo_zip: str | None = Form(None),
        supporting_materials: str | None = Form(None),
        llm_profile: str | None = Form(None),
        gpus: int = Form(0),
        time_limit: str = Form("24h"),
        image: str | None = Form(None),
        pull_policy: str | None = Form(None),
        run_final_validation: bool = Form(False),
        enable_online_research: bool = Form(False),
        objective: str = Form("paper reproduction job"),
    ) -> RedirectResponse:
        service = JobService()
        spec = _build_paper_form_spec(
            {
                "pdf_path": pdf_path,
                "paper_bundle_zip": paper_bundle_zip,
                "paper_md_path": paper_md_path,
                "inputs_zip": inputs_zip,
                "rubric_path": rubric_path,
                "blacklist_path": blacklist_path,
                "addendum_path": addendum_path,
                "submission_seed_repo_zip": submission_seed_repo_zip,
                "supporting_materials": _parse_csv_list(supporting_materials),
                "llm_profile": llm_profile or default_llm_profile_name("paper"),
                "gpus": gpus,
                "time_limit": time_limit,
                "image": image,
                "pull_policy": pull_policy,
                "run_final_validation": run_final_validation,
                "enable_online_research": enable_online_research,
                "objective": objective,
            }
        )
        job = service.create_job(spec)
        service.spawn_worker(job.id, wait=False)
        return RedirectResponse(url=f"/jobs/{job.id}", status_code=303)

    @app.post("/jobs/{job_id}/validate")
    async def validate_job(job_id: str) -> RedirectResponse:
        store = JobStore()
        job = store.get_job(job_id)
        spec = build_job_spec_clone(job, objective_suffix=" [self-check]", run_final_validation=True)
        service = JobService()
        new_job = service.create_job(spec)
        service.spawn_worker(new_job.id, wait=False)
        return RedirectResponse(url=f"/jobs/{new_job.id}", status_code=303)

    @app.post("/jobs/{job_id}/resume")
    async def resume_job(job_id: str) -> RedirectResponse:
        store = JobStore()
        job = store.get_job(job_id)
        spec = build_job_spec_clone(job, objective_suffix=" [resumed]")
        service = JobService()
        new_job = service.create_job(spec)
        service.spawn_worker(new_job.id, wait=False)
        return RedirectResponse(url=f"/jobs/{new_job.id}", status_code=303)

    @app.get("/paper/doctor", response_class=HTMLResponse)
    async def paper_doctor_page(request: Request) -> HTMLResponse:
        report = paper_doctor_report()
        return templates.TemplateResponse("doctor.html", {"request": request, "report": report})

    @app.post("/api/jobs")
    async def create_job(
        job_type: str = Form(...),
        objective: str = Form("workbench job"),
        llm_profile: str | None = Form(None),
        gpus: int = Form(0),
        time_limit: str = Form("24h"),
        image: str | None = Form(None),
        pull_policy: str | None = Form(None),
        run_final_validation: bool = Form(False),
        pdf_path: str | None = Form(None),
        paper_bundle_zip: str | None = Form(None),
        paper_md_path: str | None = Form(None),
        inputs_zip: str | None = Form(None),
        rubric_path: str | None = Form(None),
        blacklist_path: str | None = Form(None),
        addendum_path: str | None = Form(None),
        submission_seed_repo_zip: str | None = Form(None),
        supporting_materials: str | None = Form(None),
        enable_online_research: bool = Form(True),
        workspace_zip: str | None = Form(None),
        competition_bundle_zip: str | None = Form(None),
        data_dir: str | None = Form(None),
        code_repo_zip: str | None = Form(None),
        description_path: str | None = Form(None),
        sample_submission_path: str | None = Form(None),
        validation_command: str | None = Form(None),
        grading_config_path: str | None = Form(None),
        metric_direction: str | None = Form(None),
    ) -> JSONResponse:
        service = JobService()
        if job_type == "paper":
            spec = _build_paper_form_spec(
                {
                    "pdf_path": pdf_path,
                    "paper_bundle_zip": paper_bundle_zip,
                    "paper_md_path": paper_md_path,
                    "inputs_zip": inputs_zip,
                    "rubric_path": rubric_path,
                    "blacklist_path": blacklist_path,
                    "addendum_path": addendum_path,
                    "submission_seed_repo_zip": submission_seed_repo_zip,
                    "supporting_materials": _parse_csv_list(supporting_materials),
                    "llm_profile": llm_profile or default_llm_profile_name("paper"),
                    "gpus": gpus,
                    "time_limit": time_limit,
                    "image": image,
                    "pull_policy": pull_policy,
                    "run_final_validation": run_final_validation,
                    "enable_online_research": enable_online_research,
                    "objective": objective,
                }
            )
        elif job_type == "mle":
            spec = _build_mle_form_spec(
                {
                    "workspace_zip": workspace_zip,
                    "competition_bundle_zip": competition_bundle_zip,
                    "data_dir": data_dir,
                    "code_repo_zip": code_repo_zip,
                    "description_path": description_path,
                    "sample_submission_path": sample_submission_path,
                    "validation_command": validation_command,
                    "grading_config_path": grading_config_path,
                    "metric_direction": metric_direction,
                    "llm_profile": llm_profile or default_llm_profile_name("mle"),
                    "gpus": gpus,
                    "time_limit": time_limit,
                    "image": image,
                    "pull_policy": pull_policy,
                    "run_final_validation": run_final_validation,
                    "objective": objective,
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"unsupported job_type={job_type}")
        job = service.create_job(spec)
        pid = service.spawn_worker(job.id, wait=False)
        return JSONResponse({"job_id": job.id, "worker_pid": pid})

    @app.get("/api/jobs")
    async def api_jobs() -> JSONResponse:
        jobs = [job.model_dump(mode="json") for job in JobStore().list_jobs()]
        return JSONResponse(jobs)

    @app.get("/api/jobs/{job_id}")
    async def api_job(job_id: str) -> JSONResponse:
        store = JobStore()
        job = store.get_job(job_id)
        return JSONResponse(
            {
                **job.model_dump(mode="json"),
                "events": [event.model_dump(mode="json") for event in store.list_events(job_id)],
                "artifacts": [artifact.model_dump(mode="json") for artifact in store.list_artifacts(job_id)],
            }
        )

    @app.get("/api/jobs/{job_id}/artifacts")
    async def api_artifacts(job_id: str) -> JSONResponse:
        artifacts = [artifact.model_dump(mode="json") for artifact in JobStore().list_artifacts(job_id)]
        return JSONResponse(artifacts)

    @app.get("/api/jobs/{job_id}/events")
    async def api_events(job_id: str, after_id: int = 0) -> JSONResponse:
        events = [event.model_dump(mode="json") for event in JobStore().list_events(job_id, after_id=after_id)]
        return JSONResponse(events)

    @app.get("/api/jobs/{job_id}/events/stream")
    async def stream_events(job_id: str) -> StreamingResponse:
        async def event_stream():
            store = JobStore()
            last_seen = 0
            while True:
                events = store.list_events(job_id, after_id=last_seen)
                for event in events:
                    last_seen = event.id
                    payload = json.dumps(event.model_dump(mode="json"))
                    yield f"data: {payload}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/download")
    async def download_job(job_id: str) -> FileResponse:
        path = JobService().export_bundle(job_id)
        return FileResponse(path, filename=path.name, media_type="application/zip")

    @app.get("/artifacts/{job_id}/{artifact_path:path}")
    async def open_artifact(job_id: str, artifact_path: str):
        paths = ensure_job_dirs(resolve_job_paths(job_id))
        candidate = (paths.root / artifact_path).resolve()
        if not candidate.is_file() or not _within_root(paths.root, candidate):
            raise HTTPException(status_code=404, detail="artifact not found")
        if candidate.suffix.lower() in {".md", ".txt", ".json", ".jsonl", ".log", ".sh"}:
            content = candidate.read_text(encoding="utf-8", errors="replace")
            return HTMLResponse(f"<pre>{content}</pre>")
        return FileResponse(candidate)

    return app
