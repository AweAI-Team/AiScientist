from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from aisci_core.models import JobPaths
from aisci_domain_paper.paper_compat import MappedShellInterface, PathMapper


@dataclass(frozen=True)
class PaperWorkspace:
    job_paths: JobPaths
    mapper: PathMapper
    paper_dir: Path
    submission_dir: Path
    agent_dir: Path
    logs_dir: Path
    analysis_dir: Path
    subagent_logs_dir: Path

    @property
    def shell(self) -> MappedShellInterface:
        return MappedShellInterface(self.job_paths.workspace_dir, self.mapper)


def build_workspace(job_paths: JobPaths) -> PaperWorkspace:
    paper_dir = job_paths.workspace_dir / "paper"
    submission_dir = job_paths.workspace_dir / "submission"
    agent_dir = job_paths.workspace_dir / "agent"
    logs_dir = job_paths.logs_dir
    analysis_dir = agent_dir / "paper_analysis"
    subagent_logs_dir = logs_dir / "subagent_logs"

    for path in (paper_dir, submission_dir, agent_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    mapper = PathMapper(
        {
            "/home/paper": paper_dir,
            "/home/submission": submission_dir,
            "/home/agent": agent_dir,
            "/home/logs": logs_dir,
            "/home/code": submission_dir,
        }
    )
    return PaperWorkspace(
        job_paths=job_paths,
        mapper=mapper,
        paper_dir=paper_dir,
        submission_dir=submission_dir,
        agent_dir=agent_dir,
        logs_dir=logs_dir,
        analysis_dir=analysis_dir,
        subagent_logs_dir=subagent_logs_dir,
    )


def ensure_submission_repo(submission_dir: Path) -> None:
    subprocess.run(["git", "init"], cwd=submission_dir, check=False, capture_output=True)
    gitignore = submission_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "\n".join(
                [
                    "# Managed by AiScientist paper loop",
                    "venv/",
                    ".venv/",
                    "__pycache__/",
                    "*.pyc",
                    "models/",
                    "data/",
                    ".pytest_cache/",
                    ".mypy_cache/",
                    ".cache/",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def extract_pdf_excerpt(pdf_path: Path, *, max_pages: int = 3, max_chars: int = 4500) -> str:
    if not pdf_path.exists():
        return "No PDF staged.\n"
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for index, page in enumerate(reader.pages[:max_pages], start=1):
        text = (page.extract_text() or "").strip()
        if text:
            chunks.append(f"## Page {index}\n\n{text[:max_chars]}")
    return "\n\n".join(chunks) if chunks else "No text could be extracted from the PDF.\n"


def list_files(root: Path, *, max_entries: int = 120) -> list[str]:
    items: list[str] = []
    if not root.exists():
        return items
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            items.append(f"- `{rel.as_posix()}`")
            if len(items) >= max_entries:
                items.append("- ...")
                break
    return items


def markdown_join(title: str, body_lines: Iterable[str]) -> str:
    return "\n".join([f"# {title}", "", *body_lines]).rstrip() + "\n"


def build_reproduce_scaffold_script(objective: str, *, extra_notes: str = "") -> str:
    notes = extra_notes.strip()
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "$REPO_DIR"\n'
        'export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"\n\n'
        "echo 'AiScientist paper reproduction scaffold'\n"
        f"echo {objective!r}\n"
        + (f"echo {notes!r}\n" if notes else "")
        + "\n"
        "echo 'submission files:' $(find . -type f | wc -l)\n"
        "git status --short || true\n"
    )
