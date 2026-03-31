from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path


IMPL_LOG_PATH = "/home/agent/impl_log.md"
EXP_LOG_PATH = "/home/agent/exp_log.md"


@dataclass(frozen=True)
class SessionInfo:
    kind: str
    index: int
    directory: Path
    title: str
    separator: str


class PaperStateManager:
    def __init__(self, *, agent_dir: Path, logs_dir: Path, subagent_logs_dir: Path) -> None:
        self.agent_dir = agent_dir
        self.logs_dir = logs_dir
        self.subagent_logs_dir = subagent_logs_dir
        self.impl_log_path = agent_dir / "impl_log.md"
        self.exp_log_path = agent_dir / "exp_log.md"

    def ensure_logs(self) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, kind: str, *, label: str | None = None) -> SessionInfo:
        self.subagent_logs_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._normalize_kind(kind)
        existing = sorted(self.subagent_logs_dir.glob(f"{prefix}_*"))
        index = len(existing) + 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        directory = self.subagent_logs_dir / f"{prefix}_{index:03d}_{timestamp}"
        directory.mkdir(parents=True, exist_ok=True)
        title = label or self._default_label(prefix, index)
        separator = f"\n=== {title} ===\n\n"
        return SessionInfo(kind=prefix, index=index, directory=directory, title=title, separator=separator)

    def append_separator(self, session: SessionInfo) -> None:
        target = self._log_path_for_kind(session.kind)
        self._ensure_log(target, self._title_for_log(target))
        target.write_text(target.read_text(encoding="utf-8") + session.separator, encoding="utf-8")

    def recent_impl_history(self) -> str:
        return self._recent_section(self.impl_log_path, r"^=== Implement Session")

    def recent_exp_history(self) -> str:
        return self._recent_section(self.exp_log_path, r"^=== Experiment Session")

    def append_session_note(self, kind: str, summary: str, details: str = "") -> None:
        target = self._log_path_for_kind(kind)
        self._ensure_log(target, self._title_for_log(target))
        lines = [f"## [{time.strftime('%Y-%m-%d %H:%M:%S')}] {summary}", ""]
        if details:
            lines.extend([details.strip(), ""])
        lines.extend(["---", ""])
        target.write_text(target.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")

    def _recent_section(self, path: Path, pattern: str) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
        if not matches:
            return text[-6_000:].strip()
        start = matches[-1].start()
        return text[start:].strip()

    def _ensure_log(self, path: Path, title: str) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {title}\n\n---\n\n", encoding="utf-8")

    def _log_path_for_kind(self, kind: str) -> Path:
        normalized = self._normalize_kind(kind)
        if normalized == "impl":
            return self.impl_log_path
        return self.exp_log_path

    @staticmethod
    def _normalize_kind(kind: str) -> str:
        normalized = kind.strip().lower()
        if normalized in {"implementation", "implement"}:
            return "impl"
        if normalized in {"experiment", "exp"}:
            return "exp"
        if normalized in {"clean_validation", "clean_val", "validation"}:
            return "clean_val"
        return normalized

    @staticmethod
    def _default_label(kind: str, index: int) -> str:
        if kind == "impl":
            return f"Implement Session {index}"
        if kind == "exp":
            return f"Experiment Session {index}"
        if kind == "clean_val":
            return f"Experiment Session (Clean Validation {index})"
        return f"{kind.replace('_', ' ').title()} Session {index}"

    @staticmethod
    def _title_for_log(path: Path) -> str:
        return "Implementation Log" if path.name == "impl_log.md" else "Experiment Log"
