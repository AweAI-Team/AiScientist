from __future__ import annotations

import csv
import json
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aisci_app.presentation import paper_artifact_hints
from aisci_core.models import JobRecord, JobStatus
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_core.store import JobStore

DEFAULT_REFRESH_SECONDS = 2.0
GPU_REFRESH_SECONDS = 1.0
MASCOT_FRAME_SECONDS = 2.0
EVENT_LIMIT = 8
PREVIEW_LINES = 18
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
]
DETAIL_TABS = ("overview", "events", "logs", "conversation")
STATUS_STYLES = {
    JobStatus.PENDING: "yellow",
    JobStatus.RUNNING: "cyan",
    JobStatus.SUCCEEDED: "green",
    JobStatus.FAILED: "bold red",
    JobStatus.CANCELLED: "magenta",
}

PHASE_LABELS = {
    "ingest": "ingest",
    "analyze": "analyze",
    "prioritize": "prioritize",
    "implement": "implement",
    "validate": "validate",
    "finalize": "finalize",
}

PHASE_STYLES = {
    "ingest": "bright_black",
    "analyze": "yellow",
    "prioritize": "magenta",
    "implement": "cyan",
    "validate": "green",
    "finalize": "bright_white",
}

MASCOT_CLASSIC_FACES = (
    "○ ◡ ○",
    "- ﹏ -",
    "> ◡ <",
    "◕ ◡ ◕",
    "○ ◠ ○",
    "- ︶ -",
    "> ◠ <",
    "◕ ◠ ◕",
)

MASCOT_FACES = {
    "idle": MASCOT_CLASSIC_FACES,
    "thinking": (
        "◔ ◡ ◔",
        "◔ ◠ ◔",
        "◕ ﹏ ◕",
        "○ ◠ ○",
        "- ﹏ -",
        "> ◡ <",
        "◕ ◡ ◕",
        "○ ◡ ○",
    ),
    "running": (
        "◉ ◡ ◉",
        "◉ ◠ ◉",
        "> ◡ <",
        "> ◠ <",
        "◕ ◡ ◕",
        "○ ◡ ○",
        "- ︶ -",
        "◕ ◠ ◕",
    ),
    "success": (
        "● ◡ ●",
        "● ◠ ●",
        "^ ◡ ^",
        "^ ◠ ^",
        "○ ◡ ○",
        "> ◡ <",
        "◕ ◡ ◕",
        "- ︶ -",
    ),
    "error": (
        "× _ ×",
        "× ︵ ×",
        "x _ x",
        "x ︵ x",
        "- ﹏ -",
        "○ ◠ ○",
    ),
}

MASCOT_FRAMES: dict[str, list[list[str]]] = {
    "idle": [
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|o o|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | ... |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|- -|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | ... |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
    ],
    "analyze": [
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|o O|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | ?.. |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|O o|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | ..? |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
    ],
    "implement": [
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|o o|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | >_  |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|o o|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | >   |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
    ],
    "validate": [
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|^ ^|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | >_  |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/v\\_\\._.'     ",
        ],
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|^ ^|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | >   |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/v\\_\\._.'     ",
        ],
    ],
    "finalize": [
        [
            "        .-*- .        ",
            "    .--/___\\--.       ",
            "  .'/[]|^ ^|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | *** |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
        [
            "        .-*-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|^ ^|[]\\`.     ",
            " / /   | u |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | *** |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
    ],
    "failed": [
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|x x|[]\\`.     ",
            " / /   | _ |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   | !!  |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
        [
            "        .-o-.         ",
            "    .--/___\\--.       ",
            "  .'/[]|x x|[]\\`.     ",
            " / /   | _ |   \\ \\    ",
            "| |   ._____.   | |   ",
            "| |   |  !! |   | |   ",
            " \\ \\  '-----'  / /    ",
            "  '._./_/ \\_\\._.'     ",
        ],
    ],
}


@dataclass(frozen=True)
class GpuStat:
    index: str
    name: str
    utilization: int | None
    memory_used: int | None
    memory_total: int | None
    temperature: int | None


@dataclass(frozen=True)
class JobRow:
    job: JobRecord
    display_phase: str
    latest_event: str
    validation_status: str | None
    self_check_status: str | None
    gpu_binding: str


@dataclass(frozen=True)
class JobDetail:
    row: JobRow
    main_step: int | None
    recent_events: list[dict[str, Any]]
    session_state: dict[str, Any]
    validation: dict[str, Any]
    self_check: dict[str, Any]
    sandbox: dict[str, Any]
    log_previews: dict[str, str]
    artifact_lines: list[str]
    artifact_tree: str
    conversation_view: str
    gpu_stats: list[GpuStat]
    gpu_error: str | None
    subagent_counts: list[tuple[str, int]]


@dataclass(frozen=True)
class DashboardSnapshot:
    jobs: list[JobRow]
    selected_index: int
    detail: JobDetail | None
    collected_at: float

    @property
    def selected(self) -> JobRow | None:
        if not self.jobs:
            return None
        return self.jobs[self.selected_index]


@dataclass(frozen=True)
class TUIRunResult:
    job_id: str | None
    completed: bool
    detached: bool


def parse_nvidia_smi_csv(text: str) -> list[GpuStat]:
    rows = []
    for raw_row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(raw_row) < 6:
            continue
        index, name, util, mem_used, mem_total, temperature = [item.strip() for item in raw_row[:6]]
        rows.append(
            GpuStat(
                index=index,
                name=name,
                utilization=_parse_int(util),
                memory_used=_parse_int(mem_used),
                memory_total=_parse_int(mem_total),
                temperature=_parse_int(temperature),
            )
        )
    return rows


def query_nvidia_smi(command: list[str] | None = None) -> tuple[list[GpuStat], str | None]:
    try:
        completed = subprocess.run(
            command or GPU_QUERY,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found"
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "nvidia-smi failed").strip()
        return [], detail
    return parse_nvidia_smi_csv(completed.stdout), None


def run_tui_dashboard(
    *,
    job_id: str | None = None,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    once: bool = False,
    exit_when_job_done: bool = False,
    store: JobStore | None = None,
    console: Console | None = None,
) -> TUIRunResult:
    app = _DashboardApp(
        store=store or JobStore(),
        console=console or Console(),
        refresh_seconds=refresh_seconds,
        once=once,
        initial_job_id=job_id,
        exit_when_job_done=exit_when_job_done,
    )
    return app.run()


class _DashboardApp:
    def __init__(
        self,
        *,
        store: JobStore,
        console: Console,
        refresh_seconds: float,
        once: bool,
        initial_job_id: str | None,
        exit_when_job_done: bool,
    ) -> None:
        self.store = store
        self.console = console
        self.refresh_seconds = max(refresh_seconds, 0.2)
        self.once = once
        self.initial_job_id = initial_job_id
        self.exit_when_job_done = exit_when_job_done
        self.page = "detail" if initial_job_id else "jobs"
        self.detail_tab = "overview"
        self.message = ""
        self._last_gpu_poll = 0.0
        self._gpu_stats: list[GpuStat] = []
        self._gpu_error: str | None = None
        self._selected_index = 0
        self._job_ids: list[str] = []
        self._focus_job_id = initial_job_id

    def run(self) -> TUIRunResult:
        snapshot = self._build_snapshot()
        if self.once:
            self.console.print(self._render_static(snapshot))
            return self._result_from_snapshot(snapshot, detached=False)

        if not self.console.is_terminal or not sys.stdin.isatty():
            raise RuntimeError("Interactive TUI requires a TTY. Use --once for non-interactive rendering.")

        detached = False
        finished = False
        with _TerminalKeys() as keys, Live(
            self._render(snapshot),
            console=self.console,
            screen=True,
            auto_refresh=False,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            next_refresh = 0.0
            while True:
                now = time.monotonic()
                if now >= next_refresh:
                    snapshot = self._build_snapshot()
                    live.update(self._render(snapshot), refresh=True)
                    next_refresh = now + self.refresh_seconds
                    if self.exit_when_job_done and self._attached_job_finished(snapshot):
                        finished = True
                        self.message = f"Job {self.initial_job_id} finished with status={snapshot.selected.job.status.value}."
                        live.update(self._render(snapshot), refresh=True)
                        time.sleep(0.6)
                        break

                key = keys.read_key(timeout=0.1)
                if key is None:
                    continue
                if key == "q":
                    detached = bool(self.initial_job_id and not self._attached_job_finished(snapshot))
                    break
                if self._handle_key(key, snapshot):
                    snapshot = self._build_snapshot()
                    live.update(self._render(snapshot), refresh=True)
                    next_refresh = time.monotonic() + self.refresh_seconds

        return self._result_from_snapshot(snapshot, detached=detached and not finished)

    def _build_snapshot(self) -> DashboardSnapshot:
        jobs = [self._build_row(job) for job in self.store.list_jobs()]
        self._job_ids = [row.job.id for row in jobs]
        if self._focus_job_id and self._focus_job_id in self._job_ids:
            self._selected_index = self._job_ids.index(self._focus_job_id)
        elif self._selected_index >= len(self._job_ids):
            self._selected_index = max(len(self._job_ids) - 1, 0)
        elif jobs:
            self._focus_job_id = jobs[self._selected_index].job.id

        stats, error = self._current_gpu_metrics()
        detail = None
        if jobs:
            selected = jobs[self._selected_index]
            detail = self._build_detail(selected, stats, error)
        return DashboardSnapshot(
            jobs=jobs,
            selected_index=self._selected_index,
            detail=detail,
            collected_at=time.time(),
        )

    def _build_row(self, job: JobRecord) -> JobRow:
        paths = resolve_job_paths(job.id)
        validation = _read_json(paths.artifacts_dir / "validation_report.json")
        self_check = _read_json(paths.workspace_dir / "agent" / "final_self_check.json")
        latest_event = _latest_event_summary(job)
        display_phase = _latest_phase(paths.logs_dir / "conversation.jsonl", fallback=job.phase.value)
        if not latest_event and job.error:
            latest_event = _compact_text(job.error)
        return JobRow(
            job=job,
            display_phase=display_phase,
            latest_event=latest_event or "-",
            validation_status=_string_value(validation.get("status")),
            self_check_status=_string_value(self_check.get("status")),
            gpu_binding=_gpu_binding(job),
        )

    def _build_detail(
        self,
        row: JobRow,
        gpu_stats: list[GpuStat],
        gpu_error: str | None,
    ) -> JobDetail:
        job = row.job
        paths = ensure_job_dirs(resolve_job_paths(job.id))
        session_state = _read_json(paths.logs_dir / "paper_session_state.json")
        validation = _read_json(paths.artifacts_dir / "validation_report.json")
        self_check = _read_json(paths.workspace_dir / "agent" / "final_self_check.json")
        sandbox = _read_json(paths.state_dir / "sandbox_session.json")
        artifact_lines = []
        for hint in paper_artifact_hints(job):
            if hint.exists:
                artifact_lines.append(f"{hint.label}: {Path(hint.path).name}")
        artifact_lines = artifact_lines[:8]
        selected_gpu_ids = list(job.runtime_profile.gpu_ids)
        selected_gpu_stats = [stat for stat in gpu_stats if stat.index in selected_gpu_ids]
        return JobDetail(
            row=row,
            main_step=_latest_step(paths.logs_dir / "conversation.jsonl"),
            recent_events=_recent_events(job),
            session_state=session_state,
            validation=validation,
            self_check=self_check,
            sandbox=sandbox,
            log_previews={
                "job.log": _tail_text(paths.logs_dir / "job.log", PREVIEW_LINES),
                "agent.log": _tail_text(paths.logs_dir / "agent.log", PREVIEW_LINES),
            },
            artifact_lines=artifact_lines,
            artifact_tree=_workspace_tree_text(paths.workspace_dir, depth=3),
            conversation_view=_conversation_view_text(paths.logs_dir / "conversation.jsonl", limit=24),
            gpu_stats=selected_gpu_stats,
            gpu_error=gpu_error,
            subagent_counts=_collect_subagent_counts(paths.logs_dir / "conversation.jsonl"),
        )

    def _current_gpu_metrics(self) -> tuple[list[GpuStat], str | None]:
        now = time.monotonic()
        if now - self._last_gpu_poll < GPU_REFRESH_SECONDS:
            return self._gpu_stats, self._gpu_error
        self._gpu_stats, self._gpu_error = query_nvidia_smi()
        self._last_gpu_poll = now
        return self._gpu_stats, self._gpu_error

    def _attached_job_finished(self, snapshot: DashboardSnapshot) -> bool:
        selected = snapshot.selected
        if selected is None or self.initial_job_id is None:
            return False
        if selected.job.id != self.initial_job_id:
            return False
        return selected.job.status not in {JobStatus.PENDING, JobStatus.RUNNING}

    def _result_from_snapshot(self, snapshot: DashboardSnapshot, *, detached: bool) -> TUIRunResult:
        selected = snapshot.selected
        completed = bool(selected and selected.job.status not in {JobStatus.PENDING, JobStatus.RUNNING})
        return TUIRunResult(
            job_id=selected.job.id if selected else self.initial_job_id,
            completed=completed,
            detached=detached,
        )

    def _handle_key(self, key: str, snapshot: DashboardSnapshot) -> bool:
        jobs = snapshot.jobs
        if key in {"j", "down"} and jobs:
            self._selected_index = min(self._selected_index + 1, len(jobs) - 1)
            self._focus_job_id = jobs[self._selected_index].job.id
            return True
        if key in {"k", "up"} and jobs:
            self._selected_index = max(self._selected_index - 1, 0)
            self._focus_job_id = jobs[self._selected_index].job.id
            return True
        if key == "enter" and jobs:
            self.page = "detail"
            return True
        if key == "b":
            if self.page == "gpu":
                self.page = "detail" if self._focus_job_id else "jobs"
            elif self.page == "detail" and not self.initial_job_id:
                self.page = "jobs"
            return True
        if key in {"1", "2", "3", "4"} and self.page in {"detail", "gpu"}:
            self.detail_tab = DETAIL_TABS[int(key) - 1]
            if self.page == "gpu":
                self.page = "detail"
            return True
        if key == "g":
            self.page = "gpu"
            return True
        if key == "r":
            self._last_gpu_poll = 0.0
            self.message = "Refreshed."
            return True
        return False

    def _render(self, snapshot: DashboardSnapshot) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(self._render_header(snapshot), name="header", size=5),
            Layout(name="body"),
            Layout(self._render_footer(snapshot), name="footer", size=2),
        )
        if self.page == "gpu":
            layout["body"].update(self._render_gpu_page(snapshot))
        elif self.page == "detail":
            layout["body"].update(self._render_detail_page(snapshot))
        else:
            layout["body"].update(self._render_jobs_page(snapshot))
        return layout

    def _render_static(self, snapshot: DashboardSnapshot) -> RenderableType:
        if self.page == "gpu":
            body = self._render_gpu_page(snapshot)
        elif self.page == "detail":
            body = self._render_detail_page(snapshot)
        else:
            body = self._render_jobs_page(snapshot)
        return Group(
            self._render_header(snapshot),
            body,
            self._render_footer(snapshot),
        )

    def _render_header(self, snapshot: DashboardSnapshot) -> RenderableType:
        selected = snapshot.selected
        title = Text("AiScientist", style="bold bright_white")
        title.append("  ")
        title.append(self._render_mascot(selected.job if selected else None, phase_override=selected.display_phase if selected else None))

        subtitle = Text()
        subtitle.append(f"jobs={len(snapshot.jobs)}", style="white")
        if selected:
            subtitle.append("  ")
            subtitle.append(f"job={_short_job_id(selected.job.id, width=20)}", style="cyan")
            subtitle.append("  ")
            subtitle.append_text(_status_badge(selected.job.status))
            subtitle.append("  ")
            subtitle.append(_labelize_phase(selected.display_phase), style=PHASE_STYLES.get(selected.display_phase, "white"))
        if self.message:
            subtitle.append("  ")
            subtitle.append(self.message, style="yellow")
        return Panel(Group(title, subtitle), box=box.ROUNDED, border_style="cyan")

    def _render_footer(self, snapshot: DashboardSnapshot) -> RenderableType:
        hints = "j/k move  enter details  b back  1-4 tabs  g gpu  r refresh  q quit"
        if self.page == "gpu":
            hints = "b back  r refresh  q quit"
        if not snapshot.jobs:
            hints = "q quit"
        return Panel(Text(hints, style="dim"), box=box.SQUARE, border_style="bright_black")

    def _render_jobs_page(self, snapshot: DashboardSnapshot) -> RenderableType:
        if not snapshot.jobs:
            return Panel("No jobs found. Start one with `aisci paper run --wait --tui`.", box=box.ROUNDED, border_style="bright_black")

        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_lines=False, header_style="bold bright_white", row_styles=["", "dim"])
        table.add_column("", width=2)
        table.add_column("Job", style="cyan", no_wrap=True)
        table.add_column("Type", width=6)
        table.add_column("Status", width=12, no_wrap=True)
        table.add_column("Phase", width=11)
        table.add_column("Age", width=10)
        table.add_column("GPU", width=12)
        table.add_column("Checks", width=24)
        table.add_column("Latest Event", overflow="fold")
        for index, row in enumerate(snapshot.jobs):
            selected = index == snapshot.selected_index
            checks = _checks_status(row.validation_status, row.self_check_status)
            table.add_row(
                Text(">" if selected else " ", style="bold cyan" if selected else "dim"),
                row.job.id,
                row.job.job_type.value,
                _status_badge(row.job.status),
                Text(_labelize_phase(row.display_phase), style=PHASE_STYLES.get(row.display_phase, "white")),
                Text(_human_duration(row.job.duration_seconds), style="white"),
                row.gpu_binding,
                checks,
                Text(_crop(row.latest_event, 80), style="white"),
                style="bold" if selected else "",
            )

        right = []
        if snapshot.detail is not None:
            right.append(self._render_selected_summary(snapshot.detail))

        if self.console.size.width >= 120 and right:
            return Columns(
                [
                    Panel(table, title="Jobs", box=box.ROUNDED, border_style="cyan"),
                    Group(*right),
                ],
                expand=True,
                equal=False,
            )
        return Group(Panel(table, title="Jobs", box=box.ROUNDED, border_style="cyan"), *right)

    def _render_selected_summary(self, detail: JobDetail) -> RenderableType:
        job = detail.row.job
        summary = Table.grid(expand=True)
        summary.add_column(style="cyan", ratio=1)
        summary.add_column(ratio=2)
        summary.add_row("objective", _crop(job.objective, 120))
        summary.add_row("status", _status_badge(job.status))
        summary.add_row("phase", Text(_labelize_phase(detail.row.display_phase), style=PHASE_STYLES.get(detail.row.display_phase, "white")))
        summary.add_row("checks", _checks_status(detail.row.validation_status, detail.row.self_check_status))
        summary.add_row("latest", _crop(detail.row.latest_event, 180))
        summary.add_row("mode", _job_mode(job))
        if detail.main_step is not None:
            summary.add_row("orchestrator step", str(detail.main_step))
        if job.runtime_profile.gpu_ids:
            summary.add_row("gpu_ids", ", ".join(job.runtime_profile.gpu_ids))
        elif job.runtime_profile.gpu_count > 0:
            summary.add_row("gpu_count", str(job.runtime_profile.gpu_count))
        return Panel(summary, title="Selected Job", box=box.ROUNDED, border_style=_status_style(job.status.value))

    def _render_detail_page(self, snapshot: DashboardSnapshot) -> RenderableType:
        detail = snapshot.detail
        if detail is None:
            return Panel("Job not found.", box=box.ROUNDED, border_style="red")

        main = self._render_detail_main(detail)
        sidebar = self._render_detail_sidebar(detail)
        if sidebar is None:
            return main
        if self.console.size.width >= 140:
            return Columns([main, sidebar], expand=True, equal=False)
        return Group(main, sidebar)

    def _render_detail_main(self, detail: JobDetail) -> RenderableType:
        tabs = []
        for tab in DETAIL_TABS:
            style = "bold cyan" if tab == self.detail_tab else "dim"
            tabs.append(Text(f"[{DETAIL_TABS.index(tab) + 1}] {tab}", style=style))
        tab_line = Text.assemble(*sum(([item, Text("  ")] for item in tabs), [])[:-1])
        body: RenderableType
        if self.detail_tab == "overview":
            body = self._render_overview(detail)
        elif self.detail_tab == "events":
            body = self._render_events(detail)
        elif self.detail_tab == "logs":
            body = self._render_logs(detail)
        else:
            body = self._render_conversation(detail)
        return Group(
            Panel(tab_line, title=f"Job {detail.row.job.id}", box=box.ROUNDED, border_style="cyan"),
            body,
        )

    def _render_detail_sidebar(self, detail: JobDetail) -> RenderableType | None:
        _ = detail
        if self.detail_tab == "overview":
            return self._render_gpu_summary(detail)
        return None

    def _render_overview(self, detail: JobDetail) -> RenderableType:
        job = detail.row.job
        overview = Table.grid(expand=True)
        overview.add_column(style="cyan", ratio=1)
        overview.add_column(ratio=2)
        overview.add_row("status", _status_badge(job.status))
        overview.add_row("phase", Text(_labelize_phase(detail.row.display_phase), style=PHASE_STYLES.get(detail.row.display_phase, "white")))
        overview.add_row("llm", job.llm_profile)
        overview.add_row("gpu", detail.row.gpu_binding)
        overview.add_row("mode", _job_mode(job))
        overview.add_row("duration", Text(_human_duration(job.duration_seconds), style="white"))
        overview.add_row("orchestrator step", str(detail.main_step or "-"))
        overview.add_row("latest activity", _crop(detail.row.latest_event, 120))
        subagents = Table.grid(expand=True)
        subagents.add_column(style="cyan", ratio=2)
        subagents.add_column(justify="right")
        if detail.subagent_counts:
            for name, count in detail.subagent_counts:
                subagents.add_row(_labelize_subagent(name), str(count))
        else:
            subagents.add_row("activity", "no subagent runs yet")
        runtime = Table.grid(expand=True)
        runtime.add_column(style="cyan", ratio=1)
        runtime.add_column(ratio=2)
        runtime.add_row("image", _string_value(detail.sandbox.get("image_ref")) or _string_value(job.runtime_profile.image) or "-")
        runtime.add_row("container", _string_value(detail.sandbox.get("container_name")) or "-")
        return Group(
            Panel(overview, title="Overview", box=box.ROUNDED, border_style="cyan"),
            Panel(subagents, title="Subagent Calls", box=box.ROUNDED, border_style="magenta"),
            Panel(runtime, title="Runtime", box=box.ROUNDED, border_style="yellow"),
        )

    def _render_events(self, detail: JobDetail) -> RenderableType:
        body = _render_recent_events(detail.recent_events)
        artifacts_panel = Panel(
            Text(detail.artifact_tree, style="bright_black"),
            title="Artifacts",
            box=box.ROUNDED,
            border_style="yellow",
        )
        events_panel = Panel(body, title="Recent Events", box=box.ROUNDED, border_style="cyan")
        return _group_or_columns([artifacts_panel, events_panel], width=self.console.size.width, threshold=140)

    def _render_logs(self, detail: JobDetail) -> RenderableType:
        panels = []
        for name, preview in detail.log_previews.items():
            panels.append(_render_log_panel(name, preview))
        return Group(*panels)

    def _render_conversation(self, detail: JobDetail) -> RenderableType:
        return Panel(detail.conversation_view, title="Conversation", box=box.ROUNDED, border_style="cyan")

    def _render_gpu_summary(self, detail: JobDetail) -> RenderableType:
        job = detail.row.job
        if not job.runtime_profile.gpu_ids:
            if job.runtime_profile.gpu_count > 0:
                text = f"Requested {job.runtime_profile.gpu_count} GPU(s).\nUse --gpu-ids for per-device telemetry."
            else:
                text = "No GPU binding recorded for this job."
            return Panel(text, title="GPU", box=box.ROUNDED, border_style="blue")

        if detail.gpu_error:
            text = f"gpu_ids: {', '.join(job.runtime_profile.gpu_ids)}\ntelemetry unavailable: {detail.gpu_error}"
            return Panel(text, title="GPU", box=box.ROUNDED, border_style="red")

        if not detail.gpu_stats:
            text = f"gpu_ids: {', '.join(job.runtime_profile.gpu_ids)}\nwaiting for GPU telemetry."
            return Panel(text, title="GPU", box=box.ROUNDED, border_style="blue")

        panels = []
        for stat in detail.gpu_stats:
            grid = Table.grid(expand=True)
            grid.add_column(style="cyan", ratio=1)
            grid.add_column(ratio=2)
            grid.add_row("name", stat.name)
            grid.add_row("util", Text.assemble(_bar_text(stat.utilization, color="cyan"), Text(f" {stat.utilization or 0}%", style="white")))
            mem_percent = _memory_percent(stat)
            grid.add_row("memory", Text.assemble(_bar_text(mem_percent, color="magenta"), Text(f" {_memory_text(stat)}", style="white")))
            grid.add_row("temp", f"{stat.temperature if stat.temperature is not None else '-'} C")
            panels.append(
                Panel(grid, title=f"GPU {stat.index}", box=box.ROUNDED, border_style="blue")
            )
        return Group(*panels)

    def _render_gpu_page(self, snapshot: DashboardSnapshot) -> RenderableType:
        detail = snapshot.detail
        if detail is None:
            return Panel("Job not found.", box=box.ROUNDED, border_style="red")
        return Group(
            Panel(
                f"job: {detail.row.job.id}\nstatus: {detail.row.job.status.value}\ngpu: {detail.row.gpu_binding}",
                title="GPU Scope",
                box=box.ROUNDED,
                border_style="cyan",
            ),
            self._render_gpu_summary(detail),
        )

    def _render_mascot(self, job: JobRecord | None, *, phase_override: str | None = None) -> Text:
        phase = _mascot_phase(job, phase_override=phase_override)
        faces = MASCOT_FACES.get(phase, MASCOT_FACES["idle"])
        frame_index = int(time.monotonic() / MASCOT_FRAME_SECONDS) % len(faces)
        style = {
            "idle": "bright_white",
            "thinking": "yellow",
            "running": "cyan",
            "success": "green",
            "error": "bold red",
        }.get(phase, "bright_white")
        return Text(faces[frame_index], style=style)


class _TerminalKeys:
    def __enter__(self) -> "_TerminalKeys":
        self.fd = sys.stdin.fileno()
        self.previous = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)

    def read_key(self, timeout: float) -> str | None:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        char = sys.stdin.read(1)
        if char in {"\r", "\n"}:
            return "enter"
        if char == "\x1b":
            sequence = self._read_escape_sequence()
            return {
                "\x1b[A": "up",
                "\x1b[B": "down",
            }.get(sequence)
            return None
        return char

    def _read_escape_sequence(self) -> str:
        sequence = "\x1b"
        for _ in range(16):
            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not ready:
                break
            char = sys.stdin.read(1)
            sequence += char
            if char.isalpha() or char == "~":
                break
        return sequence


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "[N/A]":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return {}


def _load_recent_jsonl(path: Path, *, limit: int = EVENT_LIMIT) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    raw = _tail_bytes(path)
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records[-limit:]


def _tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return "[missing]"
    raw = _tail_bytes(path)
    text = raw.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _tail_bytes(path: Path, max_bytes: int = 65536) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        start = max(size - max_bytes, 0)
        handle.seek(start)
        data = handle.read()
    if start > 0:
        _, _, data = data.partition(b"\n")
    return data


def _latest_event_summary(job: JobRecord) -> str:
    paths = resolve_job_paths(job.id)
    records = _load_recent_jsonl(paths.logs_dir / "conversation.jsonl", limit=4)
    for record in reversed(records):
        summary = _summarize_record(record)
        if summary:
            return summary
    if job.error:
        return _compact_text(job.error)
    return ""


def _recent_events(job: JobRecord) -> list[dict[str, Any]]:
    paths = resolve_job_paths(job.id)
    records = _load_recent_jsonl(paths.logs_dir / "conversation.jsonl", limit=EVENT_LIMIT)
    events = _select_recent_feed_records(records, limit=EVENT_LIMIT)
    if events:
        return events[-EVENT_LIMIT:]
    store = JobStore()
    fallback: list[dict[str, Any]] = []
    for event in store.list_events(job.id)[-EVENT_LIMIT:]:
        fallback.append(
            {
                "event_type": "store_event",
                "phase": event.phase.value,
                "message": event.message,
            }
        )
    return fallback


def _select_recent_feed_records(records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    feed = [record for record in records if _event_belongs_in_recent_feed(record)]
    if feed:
        return feed[-limit:]
    return [record for record in records if _format_event(record)][-limit:]


def _event_belongs_in_recent_feed(record: dict[str, Any]) -> bool:
    event_kind = _string_value(record.get("event_type")) or _string_value(record.get("event")) or ""
    if event_kind in {
        "tool_result",
        "subagent_start",
        "subagent_finish",
        "status",
        "validation",
        "artifact",
        "error",
        "store_event",
    }:
        return True
    if event_kind == "model_response":
        tool_calls = record.get("tool_calls")
        return isinstance(tool_calls, list) and bool(tool_calls)
    message = _string_value(record.get("message")) or ""
    lowered = message.lower()
    return any(token in lowered for token in ("started", "finished", "completed", "failed", "timeout"))


def _latest_step(path: Path) -> int | None:
    records = _load_recent_jsonl(path, limit=20)
    steps = [record.get("step") for record in records if isinstance(record.get("step"), int)]
    return max(steps) if steps else None


def _latest_phase(path: Path, *, fallback: str) -> str:
    records = _load_recent_jsonl(path, limit=48)
    for record in reversed(records):
        phase = _phase_from_record(record)
        if phase:
            return phase
    return fallback


def _summarize_record(record: dict[str, Any]) -> str:
    message = _string_value(record.get("message"))
    if message:
        return _crop(_compact_text(message), 180)
    event_kind = _string_value(record.get("event_type")) or _string_value(record.get("event")) or "event"
    if event_kind == "model_response":
        text = _string_value(record.get("text"))
        if text:
            return _crop(_compact_text(text), 180)
        tool_calls = record.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            names = []
            for item in tool_calls:
                if isinstance(item, dict):
                    name = _string_value(item.get("name"))
                    if name:
                        names.append(name)
            if names:
                return f"Requested tools: {', '.join(names)}"
    if event_kind == "tool_result":
        tool = _string_value(record.get("tool")) or "tool"
        preview = _string_value(record.get("result_preview"))
        summary = _compact_text(preview) if preview else "completed"
        return f"{tool}: {_crop(summary, 180)}"
    return _crop(event_kind, 180)


def _format_event(record: dict[str, Any]) -> str:
    phase = _phase_from_record(record)
    step = record.get("step")
    prefix = []
    if phase:
        prefix.append(f"[{phase}]")
    if isinstance(step, int):
        prefix.append(f"step {step}")
    summary = _summarize_record(record)
    if not summary:
        return ""
    if prefix:
        return f"{' '.join(prefix)} {summary}"
    return summary


def _render_recent_events(records: list[dict[str, Any]]) -> RenderableType:
    if not records:
        return Text("No events recorded yet.", style="dim")
    lines = [_format_recent_event_text(record) for record in records]
    lines = [line for line in lines if line.plain.strip()]
    if not lines:
        return Text("No events recorded yet.", style="dim")
    return Group(*lines)


def _format_recent_event_text(record: dict[str, Any]) -> Text:
    summary = _summarize_record(record)
    if not summary:
        return Text()

    line = Text()
    step = record.get("step")
    phase = _phase_from_record(record)
    if isinstance(step, int):
        line.append(f"step {step}", style="bold cyan")
        if phase:
            line.append("  ")
    if phase:
        line.append(f"[{phase}]", style="magenta")
        line.append("  ")

    line.append(_crop(summary, 180), style=_recent_event_style(record, summary))
    return line


def _recent_event_style(record: dict[str, Any], summary: str) -> str:
    event_kind = _string_value(record.get("event_type")) or _string_value(record.get("event")) or "event"
    if event_kind == "model_response":
        return "bright_white"
    if event_kind == "tool_result":
        return "green"
    if event_kind in {"subagent_start", "subagent_finish", "store_event"}:
        return "yellow"

    lowered = summary.lower()
    if any(token in lowered for token in ("started", "finished", "completed", "failed", "timeout")):
        return "yellow"
    return "white"


def _human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(max(seconds, 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _gpu_binding(job: JobRecord) -> str:
    if job.runtime_profile.gpu_ids:
        return ",".join(job.runtime_profile.gpu_ids)
    if job.runtime_profile.gpu_count > 0:
        return f"count:{job.runtime_profile.gpu_count}"
    return "-"


def _checks_status(validation_status: str | None, self_check_status: str | None) -> Text:
    text = Text()
    text.append(f"repro:{validation_status or '-'}", style=_status_text_style(validation_status))
    text.append(" / ")
    text.append(f"review:{self_check_status or '-'}", style=_status_text_style(self_check_status))
    return text


def _status_text_style(status: str | None) -> str:
    return _status_style(status)


def _group_or_columns(
    renderables: list[RenderableType],
    *,
    width: int,
    threshold: int,
    equal: bool = False,
) -> RenderableType:
    if width >= threshold:
        return Columns(renderables, expand=True, equal=equal)
    return Group(*renderables)


def _compact_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _crop(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _bar(percent: int | None, *, width: int = 12) -> str:
    if percent is None:
        return "-" * width
    percent = max(0, min(percent, 100))
    filled = round(width * percent / 100)
    return "#" * filled + "." * (width - filled)


def _bar_text(percent: int | None, *, width: int = 12, color: str = "cyan") -> Text:
    if percent is None:
        return Text("░" * width, style="bright_black")
    percent = max(0, min(percent, 100))
    filled = round(width * percent / 100)
    text = Text()
    text.append("█" * filled, style=color)
    text.append("░" * (width - filled), style="bright_black")
    return text


def _render_log_panel(name: str, preview: str) -> Panel:
    border_style = "cyan" if name == "job.log" else "magenta"
    body = _render_log_preview_body(preview)
    subtitle = Text(f"tail {PREVIEW_LINES} lines", style="dim")
    return Panel(Group(subtitle, Text(), body), title=name, box=box.ROUNDED, border_style=border_style)


def _render_log_preview_body(preview: str) -> RenderableType:
    if not preview:
        return Text("(empty)", style="dim")
    lines = preview.splitlines()
    if not lines:
        return Text("(empty)", style="dim")
    render = Text()
    width = max(len(str(len(lines))), 2)
    for index, line in enumerate(lines, start=1):
        render.append(str(index).rjust(width), style="bright_black")
        render.append(" │ ", style="bright_black")
        style = "bright_white"
        lowered = line.lower()
        if "error" in lowered or "traceback" in lowered or "failed" in lowered:
            style = "red"
        elif "warning" in lowered or "warn" in lowered:
            style = "yellow"
        elif "success" in lowered or "passed" in lowered or "completed" in lowered:
            style = "green"
        render.append(line or " ", style=style)
        if index < len(lines):
            render.append("\n")
    return render


def _memory_percent(stat: GpuStat) -> int | None:
    if stat.memory_used is None or stat.memory_total in {None, 0}:
        return None
    return round(stat.memory_used / stat.memory_total * 100)


def _memory_text(stat: GpuStat) -> str:
    if stat.memory_used is None or stat.memory_total is None:
        return "-"
    return f"{stat.memory_used}/{stat.memory_total} MiB"


def _workspace_tree_text(path: Path, *, depth: int) -> str:
    if not path.exists():
        return "/home\n└── [missing]"

    ignored = {".git", "venv", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    lines = ["/home"]

    def walk(current: Path, prefix: str, level: int) -> None:
        if level >= depth:
            return
        children = sorted(
            [child for child in current.iterdir() if child.name not in ignored],
            key=lambda child: (child.is_file(), child.name.lower()),
        )
        for index, child in enumerate(children):
            connector = "└──" if index == len(children) - 1 else "├──"
            label = f"{child.name}/" if child.is_dir() else child.name
            lines.append(f"{prefix}{connector} {label}")
            if child.is_dir():
                extension = "    " if index == len(children) - 1 else "│   "
                walk(child, prefix + extension, level + 1)

    walk(path, "", 0)
    return "\n".join(lines)


def _conversation_view_text(path: Path, *, limit: int) -> str:
    records = _load_recent_jsonl(path, limit=limit)
    lines = [_format_conversation_record(record) for record in records]
    lines = [line for line in lines if line]
    return "\n".join(lines) if lines else "No conversation events recorded yet."


def _format_conversation_record(record: dict[str, Any]) -> str:
    summary = _conversation_record_summary(record)
    if not summary:
        return ""
    parts = []
    step = record.get("step")
    phase = _phase_from_record(record)
    if isinstance(step, int):
        parts.append(f"step {step}")
    if phase:
        parts.append(f"[{phase}]")
    prefix = " ".join(parts)
    return f"{prefix} {summary}".strip()


def _conversation_record_summary(record: dict[str, Any]) -> str:
    event_kind = _string_value(record.get("event_type")) or _string_value(record.get("event")) or "event"
    if event_kind == "model_response":
        text = _string_value(record.get("text")) or _string_value(record.get("message"))
        if text:
            return f"agent: {_crop(_compact_text(text), 180)}"
        tool_calls = record.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            names = []
            for item in tool_calls:
                if isinstance(item, dict):
                    name = _string_value(item.get("name"))
                    if name:
                        names.append(name)
            if names:
                return f"agent requested: {', '.join(names)}"
        return "agent responded."

    if event_kind == "tool_result":
        tool = _string_value(record.get("tool")) or "tool"
        preview = _string_value(record.get("result_preview"))
        if preview:
            return f"{tool}: {_crop(_compact_text(preview), 180)}"
        return f"{tool}: completed"

    message = _string_value(record.get("message"))
    if message:
        return _crop(_compact_text(message), 180)
    return _crop(event_kind, 180)


def _phase_from_record(record: dict[str, Any]) -> str | None:
    phase = _string_value(record.get("phase"))
    if phase:
        return phase
    tool = _string_value(record.get("tool"))
    if tool:
        return _phase_from_tool_name(tool)
    tool_calls = record.get("tool_calls")
    if isinstance(tool_calls, list):
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            name = _string_value(item.get("name"))
            inferred = _phase_from_tool_name(name)
            if inferred:
                return inferred
    return None


def _phase_from_tool_name(tool_name: str | None) -> str | None:
    if tool_name in {"read_paper", "read_paper_md"}:
        return "analyze"
    if tool_name == "prioritize_tasks":
        return "prioritize"
    if tool_name == "implement":
        return "implement"
    if tool_name in {"clean_reproduce_validation", "validate"}:
        return "validate"
    if tool_name == "submit":
        return "finalize"
    return None


def _collect_subagent_counts(path: Path) -> list[tuple[str, int]]:
    records = _load_recent_jsonl(path, limit=256)
    counts: dict[str, int] = {}
    for record in records:
        event_type = _string_value(record.get("event_type")) or _string_value(record.get("event"))
        if event_type != "subagent_start":
            continue
        kind = _subagent_kind(record)
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _subagent_kind(record: dict[str, Any]) -> str | None:
    payload = record.get("payload")
    if isinstance(payload, dict):
        session_dir = _string_value(payload.get("session_dir"))
        if session_dir:
            return _subagent_kind_from_session_dir(Path(session_dir).name)
    message = _string_value(record.get("message"))
    if not message:
        return None
    match = re.match(r"([a-z0-9_]+) subagent started\.", message)
    if match:
        return match.group(1)
    return None


def _subagent_kind_from_session_dir(name: str) -> str:
    match = re.match(r"(.+)_\d{3}_\d{8}_\d{6}$", name)
    if match:
        return match.group(1)
    return name


def _labelize_subagent(name: str) -> str:
    aliases = {
        "env_setup": "environment setup",
        "impl": "implementation",
    }
    if name in aliases:
        return aliases[name]
    return name.replace("_", " ")


def _labelize_phase(name: str | None) -> str:
    if not name:
        return "-"
    return PHASE_LABELS.get(name, name)


def _job_mode(job: JobRecord) -> str:
    if job.runtime_profile.workspace_layout:
        return job.runtime_profile.workspace_layout.value
    return job.job_type.value


def _mascot_phase(job: JobRecord | None, *, phase_override: str | None = None) -> str:
    if job is None:
        return "idle"
    if job.status == JobStatus.FAILED:
        return "error"
    if job.status == JobStatus.SUCCEEDED:
        return "success"
    phase = phase_override or job.phase.value
    if phase in {"analyze", "prioritize"}:
        return "thinking"
    if phase in {"implement", "validate", "finalize"}:
        return "running"
    return "idle"


def _compact_face(frame_index: int) -> str:
    _ = frame_index
    return "○ ◡ ○"


def _short_job_id(job_id: str, *, width: int = 22) -> str:
    if len(job_id) <= width:
        return job_id
    edge = max((width - 3) // 2, 6)
    return f"{job_id[:edge]}...{job_id[-edge:]}"


def _status_style(status: str | None) -> str:
    normalized = (status or "").strip().lower()
    if normalized in {"passed", "succeeded", JobStatus.SUCCEEDED.value, "ok", "available", "enabled"}:
        return "green"
    if normalized in {"failed", "fail", JobStatus.FAILED.value}:
        return "bold red"
    if normalized in {"pending", JobStatus.PENDING.value}:
        return "yellow"
    if normalized in {"running", JobStatus.RUNNING.value}:
        return "cyan"
    if normalized in {"cancelled", JobStatus.CANCELLED.value}:
        return "magenta"
    if normalized in {"skipped", "warn", "warning"}:
        return "yellow"
    return "white"


def _status_badge(status: JobStatus | str | None) -> Text:
    label = status.value if isinstance(status, JobStatus) else (status or "-")
    normalized = label.strip().lower()
    icon = {
        "passed": "●",
        "succeeded": "●",
        JobStatus.SUCCEEDED.value: "●",
        "failed": "✕",
        "fail": "✕",
        JobStatus.FAILED.value: "✕",
        "running": "◉",
        JobStatus.RUNNING.value: "◉",
        "pending": "◌",
        JobStatus.PENDING.value: "◌",
        "cancelled": "◌",
        JobStatus.CANCELLED.value: "◌",
        "skipped": "△",
    }.get(normalized, "•")
    return Text(f"{icon} {label}", style=_status_style(label))
