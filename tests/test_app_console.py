from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from aisci_app.cli import _log_targets_for_kind, app
from aisci_app.worker_main import main as worker_main
from aisci_app.presentation import paper_doctor_report, paper_job_summary
from aisci_app.tui import (
    TUIRunResult,
    _conversation_view_text,
    _select_recent_feed_records,
    _format_recent_event_text,
    _workspace_tree_text,
    parse_nvidia_smi_csv,
)
from aisci_core.models import JobRecord, JobSpec, JobStatus, JobType, PaperSpec, RunPhase, RuntimeProfile, WorkspaceLayout
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_core.store import JobStore
from aisci_app.service import JobService


def _paper_job_record(*, status: JobStatus, phase: RunPhase, error: str | None = None) -> JobRecord:
    now = datetime.now().astimezone()
    return JobRecord(
        id="paper-job-cli",
        job_type=JobType.PAPER,
        status=status,
        phase=phase,
        objective="paper cli",
        llm_profile="paper-default",
        runtime_profile=RuntimeProfile(
            workspace_layout=WorkspaceLayout.PAPER,
            run_final_validation=True,
        ),
        mode_spec=PaperSpec(pdf_path="/tmp/paper.pdf"),
        created_at=now,
        updated_at=now,
        started_at=now,
        ended_at=now,
        error=error,
    )


def _create_paper_job(tmp_path: Path):
    store = JobStore()
    service = JobService(store=store)
    job = service.create_job(
        JobSpec(
            job_type=JobType.PAPER,
            objective="paper console",
            llm_profile="paper-default",
            runtime_profile=RuntimeProfile(
                workspace_layout=WorkspaceLayout.PAPER,
                run_final_validation=True,
            ),
            mode_spec=PaperSpec(pdf_path=str(tmp_path / "paper.pdf")),
        )
    )
    paths = ensure_job_dirs(resolve_job_paths(job.id))
    (paths.logs_dir / "job.log").write_text("main log line\n", encoding="utf-8")
    (paths.logs_dir / "conversation.jsonl").write_text(
        (
            json.dumps({"event_type": "model_response", "phase": "analyze", "message": "paper analysis started"})
            + "\n"
            + json.dumps({"event_type": "subagent_start", "phase": "implement", "message": "implementation subagent started."})
            + "\n"
        ),
        encoding="utf-8",
    )
    (paths.logs_dir / "agent.log").write_text("agent log line\n", encoding="utf-8")
    (paths.logs_dir / "paper_session_state.json").write_text(
        json.dumps(
            {
                "summary": "paper summary for TUI",
                "impl_runs": 2,
                "exp_runs": 1,
                "self_checks": 1,
                "submit_attempts": 1,
                "clean_validation_called": True,
            }
        ),
        encoding="utf-8",
    )
    (paths.logs_dir / "subagent_logs").mkdir(parents=True, exist_ok=True)
    (paths.logs_dir / "subagent_logs" / "implement_001_20260331_120000").mkdir(parents=True, exist_ok=True)
    (paths.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)
    (paths.workspace_dir / "agent" / "paper_analysis").mkdir(parents=True, exist_ok=True)
    (paths.workspace_dir / "agent" / "paper_analysis" / "summary.md").write_text("# summary\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "prioritized_tasks.md").write_text("# tasks\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "plan.md").write_text("# plan\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "impl_log.md").write_text("# impl\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "exp_log.md").write_text("# exp\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "final_self_check.json").write_text(
        json.dumps({"status": "passed", "result": "self check ok"}),
        encoding="utf-8",
    )
    (paths.workspace_dir / "agent" / "final_self_check.md").write_text("# self-check\n", encoding="utf-8")
    (paths.state_dir / "resolved_llm_config.json").write_text(
        json.dumps({"profile": "paper-default", "backend": "openai"}),
        encoding="utf-8",
    )
    (paths.state_dir / "paper_main_prompt.md").write_text("# prompt\n", encoding="utf-8")
    (paths.state_dir / "capabilities.json").write_text(
        json.dumps(
            {
                "online_research": {"available": True},
                "linter": {"available": True},
            }
        ),
        encoding="utf-8",
    )
    (paths.state_dir / "sandbox_session.json").write_text(
        json.dumps({"container_name": "paper-test-session", "image_ref": "aisci-paper:test"}),
        encoding="utf-8",
    )
    (paths.artifacts_dir / "validation_report.json").write_text(
        json.dumps({"status": "passed", "summary": "validation ok"}),
        encoding="utf-8",
    )
    (paths.workspace_dir / "submission" / "reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return job, paths


def test_paper_job_summary_exposes_product_signals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    job, _ = _create_paper_job(tmp_path)
    summary = paper_job_summary(job)
    assert summary["paper_capabilities"]["online_research"] == "available"
    assert summary["paper_capabilities"]["final_self_check"] == "enabled"
    assert any(item["label"] == "main job log" for item in summary["paper_log_targets"])
    assert any(item["label"] == "sandbox session" for item in summary["paper_log_targets"])
    assert any(item["label"].startswith("session: ") for item in summary["paper_log_targets"])
    assert any(item["label"] == "paper summary" for item in summary["paper_artifacts"])
    assert any(item["label"] == "resolved llm config" for item in summary["paper_artifacts"])
    assert any(item["label"] == "main prompt snapshot" for item in summary["paper_artifacts"])


def test_log_target_helper_lists_paper_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    job, _ = _create_paper_job(tmp_path)
    targets = _log_targets_for_kind(job.id, "all")
    labels = {label for label, _ in targets}
    assert "main job log" in labels
    assert "conversation log" in labels
    assert "subagent logs" in labels


def test_paper_doctor_reports_console_tip() -> None:
    report = paper_doctor_report()
    assert any(check.name == "paper console" for check in report)
    assert any(check.name == "online_research" for check in report)


def test_parse_nvidia_smi_csv_extracts_metrics() -> None:
    rows = parse_nvidia_smi_csv("0, NVIDIA A100, 78, 10240, 40960, 63\n")
    assert rows[0].index == "0"
    assert rows[0].utilization == 78
    assert rows[0].memory_total == 40960


def test_workspace_tree_text_renders_home_style_tree(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "agent" / "paper_analysis").mkdir(parents=True)
    (workspace / "submission").mkdir(parents=True)
    (workspace / "submission" / "reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    rendered = _workspace_tree_text(workspace, depth=2)

    assert "/home" in rendered
    assert "agent/" in rendered
    assert "paper_analysis/" in rendered
    assert "submission/" in rendered
    assert "reproduce.sh" in rendered


def test_conversation_view_text_normalizes_records(tmp_path: Path) -> None:
    path = tmp_path / "conversation.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"event": "model_response", "step": 3, "phase": "implement", "text": "Working on the core implementation."}),
                json.dumps({"event": "tool_result", "step": 3, "tool": "bash", "result_preview": "pytest passed"}),
                json.dumps({"event_type": "subagent_start", "phase": "validate", "message": "experiment subagent started."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rendered = _conversation_view_text(path, limit=10)

    assert "step 3 [implement] agent: Working on the core implementation." in rendered
    assert "step 3 bash: pytest passed" in rendered
    assert "[validate] experiment subagent started." in rendered


def test_format_recent_event_text_applies_structured_prefixes() -> None:
    rendered = _format_recent_event_text(
        {
            "event_type": "subagent_start",
            "step": 7,
            "phase": "analyze",
            "message": "paper_structure subagent started.",
        }
    )

    assert rendered.plain == "step 7  [analyze]  paper_structure subagent started."
    assert any(span.style == "bold cyan" for span in rendered.spans)
    assert any(span.style == "magenta" for span in rendered.spans)
    assert any(span.style == "yellow" for span in rendered.spans)


def test_recent_feed_prefers_operational_events_over_agent_transcript() -> None:
    records = [
        {"event_type": "model_response", "phase": "implement", "text": "I will inspect the implementation details."},
        {"event_type": "tool_result", "phase": "implement", "tool": "bash", "result_preview": "pytest passed"},
        {"event_type": "subagent_start", "phase": "validate", "message": "experiment subagent started."},
    ]

    selected = _select_recent_feed_records(records, limit=10)

    assert len(selected) == 2
    assert selected[0]["event_type"] == "tool_result"
    assert selected[1]["event_type"] == "subagent_start"


def test_cli_global_env_file_option_loads_api_key(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    env_file = tmp_path / "paper.env"
    env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AISCI_PAPER_DOCTOR_PROFILE", "gpt-5.4")

    result = runner.invoke(app, ["--env-file", str(env_file), "paper", "doctor"])

    assert result.exit_code == 0
    assert "- api_key: ok (Backend credentials detected)" in result.stdout


def test_cli_global_output_root_option_sets_env(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.delenv("AISCI_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr("aisci_app.cli.paper_doctor_report", lambda: [])

    result = runner.invoke(app, ["--output-root", str(tmp_path / "runtime"), "paper", "doctor"])

    assert result.exit_code == 0
    assert os.environ["AISCI_OUTPUT_ROOT"] == str((tmp_path / "runtime").resolve())


def test_worker_main_returns_nonzero_when_job_fails(monkeypatch) -> None:
    class _Runner:
        def run_job(self, job_id: str) -> JobStatus:
            assert job_id == "job-123"
            return JobStatus.FAILED

    monkeypatch.setattr("aisci_app.worker_main.JobRunner", lambda: _Runner())
    assert worker_main(["job-123"]) == 1


def test_paper_run_wait_reports_final_failure(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    created_job = _paper_job_record(status=JobStatus.PENDING, phase=RunPhase.INGEST)
    final_job = _paper_job_record(status=JobStatus.FAILED, phase=RunPhase.ANALYZE, error="docker missing")

    class _Store:
        def get_job(self, job_id: str) -> JobRecord:
            assert job_id == created_job.id
            return final_job

    class _Service:
        def __init__(self) -> None:
            self.store = _Store()

        def create_job(self, spec) -> JobRecord:  # noqa: ANN001
            return created_job

        def spawn_worker(self, job_id: str, wait: bool = False) -> int:
            assert job_id == created_job.id
            assert wait is True
            return 1

    monkeypatch.setattr("aisci_app.cli.JobService", _Service)
    monkeypatch.setattr("aisci_app.cli.build_paper_job_spec", lambda **kwargs: object())

    result = runner.invoke(app, ["paper", "run", "--pdf", str(tmp_path / "paper.pdf"), "--wait"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["job_id"] == created_job.id
    assert payload["status"] == "failed"
    assert payload["phase"] == "analyze"
    assert payload["error"] == "docker missing"


def test_paper_run_accepts_gpu_ids(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class _Service:
        def __init__(self) -> None:
            self.store = None

        def create_job(self, spec) -> JobRecord:  # noqa: ANN001
            return _paper_job_record(status=JobStatus.PENDING, phase=RunPhase.INGEST)

        def spawn_worker(self, job_id: str, wait: bool = False) -> int:  # noqa: ARG002
            return 0

    def fake_build_paper_job_spec(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("aisci_app.cli.JobService", _Service)
    monkeypatch.setattr("aisci_app.cli.build_paper_job_spec", fake_build_paper_job_spec)

    result = runner.invoke(
        app,
        ["paper", "run", "--pdf", str(tmp_path / "paper.pdf"), "--gpu-ids", "4,5"],
    )

    assert result.exit_code == 0
    assert captured["gpus"] == 0
    assert captured["gpu_ids"] == ["4", "5"]


def test_paper_run_rejects_gpu_count_and_ids_together(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["paper", "run", "--pdf", str(tmp_path / "paper.pdf"), "--gpus", "2", "--gpu-ids", "4,5"],
    )

    assert result.exit_code != 0
    assert "Use either --gpus <count> or --gpu-ids <id,id>, not both." in (result.stdout + result.stderr)


def test_paper_run_tui_requires_wait(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["paper", "run", "--pdf", str(tmp_path / "paper.pdf"), "--detach", "--tui"],
    )

    assert result.exit_code != 0
    assert "--tui requires --wait." in (result.stdout + result.stderr)


def test_paper_run_tui_detach_emits_started_payload(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    created_job = _paper_job_record(status=JobStatus.PENDING, phase=RunPhase.INGEST)
    captured: dict[str, object] = {}

    class _Service:
        def __init__(self) -> None:
            self.store = None

        def create_job(self, spec) -> JobRecord:  # noqa: ANN001
            return created_job

        def spawn_worker(self, job_id: str, wait: bool = False) -> int:
            captured["job_id"] = job_id
            captured["wait"] = wait
            return 4242

    monkeypatch.setattr("aisci_app.cli.JobService", _Service)
    monkeypatch.setattr("aisci_app.cli.build_paper_job_spec", lambda **kwargs: object())
    monkeypatch.setattr("aisci_app.cli._is_interactive_terminal", lambda: True)
    monkeypatch.setattr(
        "aisci_app.cli._run_tui_or_exit",
        lambda **kwargs: TUIRunResult(job_id=created_job.id, completed=False, detached=True),
    )

    result = runner.invoke(
        app,
        ["paper", "run", "--pdf", str(tmp_path / "paper.pdf"), "--wait", "--tui"],
    )

    assert result.exit_code == 0
    assert captured["wait"] is False
    payload = json.loads(result.stdout)
    assert payload["job_id"] == created_job.id
    assert payload["status"] == "started"
    assert payload["worker"] == 4242


def test_tui_once_renders_jobs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setenv("AISCI_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    job, _ = _create_paper_job(tmp_path)
    monkeypatch.setattr("aisci_app.tui.query_nvidia_smi", lambda command=None: ([], "nvidia-smi unavailable"))

    result = runner.invoke(app, ["tui", "--once"])

    assert result.exit_code == 0
    assert "AiScientist" in result.stdout
    assert job.id[-8:] in result.stdout
    assert "Selected Job" in result.stdout
    assert "Terminal TUI" not in result.stdout
    assert "ai-sci" not in result.stdout


def test_tui_job_once_renders_detail(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setenv("AISCI_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    job, _ = _create_paper_job(tmp_path)
    monkeypatch.setattr("aisci_app.tui.query_nvidia_smi", lambda command=None: ([], "nvidia-smi unavailable"))

    result = runner.invoke(app, ["tui", "job", job.id, "--once"])

    assert result.exit_code == 0
    assert f"Job {job.id}" in result.stdout
    assert "Overview" in result.stdout
    assert "[4] conversation" in result.stdout
    assert "implement" in result.stdout
    assert "Subagent Calls" in result.stdout
    assert "Capabilities" not in result.stdout
    assert "validation_mode" not in result.stdout
    assert "self_check" not in result.stdout


def test_serve_command_removed() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["serve"])

    assert result.exit_code != 0
    assert "No such command 'serve'" in (result.stdout + result.stderr)
